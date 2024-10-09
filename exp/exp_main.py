from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, CATS, FITS, iTransformer, TimesNet, TiDE, FormerBaseline, Autoregressive
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.scientific_report import plot_aligned_heatmap, mts_visualize, vis_channel_forecasting, mts_visualize_horizontal

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
from collections import OrderedDict

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import logging
import copy

torch._logging.set_logs(dynamo=logging.INFO)
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.suppress_errors = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def create_directory(path):
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.writer = SummaryWriter('runs/{}_{}'.format(self.args.model_id, time.strftime("%Y%m%d-%H%M%S",time.localtime())))
        self.vali_times = 0
        self.test_times = 0
        self.steps = 0
        self.test_every = self.args.test_every
        self.early_stop = False

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'CATS': CATS,
            'FITS': FITS,
            'iTransformer': iTransformer,
            'TimesNet': TimesNet,
            'TiDE': TiDE,
            'FormerBaseline': FormerBaseline,
            'Autoregressive': Autoregressive,
        }
        model = model_dict[self.args.model].Model(self.args)
            
        if self.args.resume != 'none':
            state_dict = torch.load(self.args.resume, map_location='cpu')

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict, strict=False)
        
        if self.args.compile:
            model = torch.compile(model, mode="reduce-overhead")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        params = self.model.parameters()
        model_optim = optim.AdamW(params, lr=self.args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, label='vali'):
        total_loss = []
        preds = []
        preds_add = []
        trues = []
        self.model.eval()
        print(f'Start Validation ({label})')
        with torch.no_grad():
            for i, data in enumerate(vali_loader):
                batch_x = data[0].float().to(self.device, non_blocking=True)
                batch_y = data[1].float().to(self.device, non_blocking=True)
                batch_x_mark = data[2].float().to(self.device, non_blocking=True)
                batch_y_mark = data[3].float().to(self.device, non_blocking=True)
                
                pred_len = self.args.pred_len # batch_y.shape[1] - batch_x.shape[1]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=True)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if 'CATS' in self.args.model:
                            outputs, masks, x_predictor_snapshot = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_mask=True)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'CATS' in self.args.model:
                        outputs, masks, x_predictor_snapshot = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_mask=True)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                addtional_loss = 0
                f_dim = -self.args.number_of_targets
                if self.args.features == 'MS':
                    f_dim = -1
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                    
                preds.append(pred.float().numpy(force=True))
                trues.append(true.float().numpy(force=True))
                loss = criterion(pred, true)
                
                total_loss.append(loss)
                
        print(f'Validation ({label}): Inference Finished')
        
        total_loss = np.average(total_loss)  
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        mae_ot, mse_ot, rmse_ot, mape_ot, mspe_ot, rse_ot, corr_ot = metric(preds[:, :, -1], trues[:, :, -1])
        total_loss = mse
        
        if label == 'test':
            print(f'Validation ({label}): Visualization')
            self.writer.add_scalar(f'Loss/{label}LossAvg', float(total_loss), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMSEAvg', float(mse), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMAEAvg', float(mae), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossRMSEAvg', float(rmse), self.test_times)
            self.writer.add_scalar(f'Loss/{label}OTLossMSEAvg', float(mse_ot), self.test_times)
            self.writer.add_scalar(f'Loss/{label}OTLossMAEAvg', float(mae_ot), self.test_times)
            self.writer.add_scalar(f'Loss/{label}OTLossRMSEAvg', float(rmse_ot), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMAPEAvg', float(mape), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMSPEAvg', float(mspe), self.test_times)
            pred = pred.float().numpy()
            cbatch_x = torch.cat([batch_x[:, :, f_dim:], batch_y], dim=1).detach().cpu()
            cbatch_x = cbatch_x.float().numpy()
            met = f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, MSPE: {mspe:.4f}'
            
            if self.test_times % self.args.plot_every == 0:
                fig = mts_visualize(pred[0, :, -256:], cbatch_x[0, :, -256:], split_step=batch_x.shape[1], title=met, dpi=72, col_names=vali_data.col_names)
                if not os.path.exists("imgs"): os.makedirs("imgs")
                if not os.path.exists(f"imgs/{self.args.model_id}"): os.makedirs(f"imgs/{self.args.model_id}")
                fig.savefig(f"imgs/{self.args.model_id}/{self.test_times}.pdf", format="pdf", bbox_inches = 'tight')
                self.writer.add_figure('MTS_VS[1]', fig, self.test_times)
                plt.clf()
                if not os.path.exists("imgs_testset"): os.makedirs("imgs_testset")
                if not os.path.exists(f"imgs_testset/{self.args.model_id}"): os.makedirs(f"imgs_testset/{self.args.model_id}")

            # Additional visualization of CATS
            if 'CATS' in self.args.model:
                if self.test_times % self.args.plot_every == 0:
                    vis_true = torch.cat([batch_x[:, :, f_dim:], batch_y], dim=1)
                    if 'mid' in masks:
                        current_mask = masks['mid'][0]
                    else:
                        current_mask = torch.ones(batch_x.shape[1], x_predictor_snapshot.shape[-1])
                    fig = vis_channel_forecasting(batch_x.shape[1], x_predictor_snapshot[0][:, -256:], cum_outputs.to(x_predictor_snapshot.device)[0][:, -256:], vis_true[0][:, -256:], current_mask[:, -256:], col_names=vali_data.col_names)
                    if x_predictor_snapshot.shape[-1] < 128:
                        self.writer.add_figure(f'ChannelVis', fig, self.test_times)
                    else:
                        if not os.path.exists("imgs"): os.makedirs("imgs")
                        if not os.path.exists(f"imgs/{self.args.model_id}"): os.makedirs(f"imgs/{self.args.model_id}")
                        fig.savefig(f"imgs/{self.args.model_id}/C_{self.test_times}.pdf", format="pdf", bbox_inches = 'tight')
                    plt.clf()
                    
            self.test_times += 1
            
        if label == 'vali':
            self.writer.add_scalar(f'Loss/{label}LossAvg', float(total_loss), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMSEAvg', float(mse), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMAEAvg', float(mae), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossRMSEAvg', float(rmse), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMAPEAvg', float(mape), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMSPEAvg', float(mspe), self.vali_times)
            self.vali_times += 1
        
        self.model.train()
        print('Validation Finished')
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, configs=self.args)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            pct_start = self.args.pct_start,
                                            div_factor = 10,
                                            anneal_strategy='linear',
                                            epochs=self.args.train_epochs+1,
                                            steps_per_epoch=self.args.test_every if self.args.test_every else train_steps,
                                            max_lr = self.args.learning_rate)

        epoch_time = time.time()
        
        for epoch in range(self.args.train_epochs):
            print(f'Starting Training Epoch: {epoch}')
            iter_count = 0
            train_loss = []
            self.model.train()
            for i, data in enumerate(train_loader):
                self.steps += 1
                iter_count += 1
                model_optim.zero_grad()
                batch_x = data[0].float().to(self.device, non_blocking=True)
                batch_y = data[1].float().to(self.device, non_blocking=True)
                batch_x_mark = data[2].float().to(self.device, non_blocking=True)
                batch_y_mark = data[3].float().to(self.device, non_blocking=True)
                pred_len = self.args.pred_len # batch_y.shape[1] - batch_x.shape[1]
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=True)

                addtional_loss = torch.Tensor([0]).to(batch_x.device)

                # Random dropping implementation
                drop_mask = None
                if self.args.random_drop_training:
                    if torch.rand(1).item() > 0:
                        random_drop_rate = torch.rand(1).item()
                        drop_mask = torch.rand(1, 1, batch_x.shape[2], device=batch_x.device) < 1-random_drop_rate
                        batch_x = batch_x.masked_fill(drop_mask, 0)
                        batch_y = batch_y.masked_fill(drop_mask, 0)

                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if 'CATS' in self.args.model or 'Autoregressive' in self.args.model:
                            outputs, add_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, train=True)
                            addtional_loss += add_loss
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)     
                else:
                    if 'CATS' in self.args.model or 'Autoregressive' in self.args.model:
                        outputs, add_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, train=True)
                        addtional_loss += add_loss
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # , batch_y
                
                f_dim = -self.args.number_of_targets
                if self.args.features == 'MS':
                    f_dim = -1
                batch_y = batch_y[:, -pred_len:, f_dim:] if self.args.model != 's2s' else batch_y[:, :, f_dim:]
                
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = outputs[:, -pred_len:, f_dim:]
                        MSE = criterion(outputs, batch_y)
                        loss = MSE + addtional_loss
                else:
                    outputs = outputs[:, -pred_len:, f_dim:]
                    MSE = criterion(outputs, batch_y)
                    loss = MSE + addtional_loss
                
                self.writer.add_scalar(f'Loss/TrainLossADD', float(addtional_loss.item()), self.steps)
                self.writer.add_scalar(f'Loss/TrainLossMSE', float(MSE.item()), self.steps)
                self.writer.add_scalar(f'Loss/TrainLossTOT', float(loss.item()), self.steps)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    if self.args.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    if (iter_count + 1) % self.args.gradient_accumulation == 0:
                        scaler.step(model_optim)
                        scaler.update()
                        model_optim.zero_grad()
                        
                else:
                    loss.backward()
                    if self.args.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    if (iter_count + 1) % self.args.gradient_accumulation == 0:
                        model_optim.step()
                        model_optim.zero_grad()
                
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                if self.test_every != 0 and self.steps % self.test_every == 0:
                    print("Test Steps: {} cost time: {}".format(self.test_every, time.time() - epoch_time))
                    self.writer.add_scalar(f'LR/LearningRate', float(scheduler.get_last_lr()[0]), self.vali_times)
                    tl = np.average(train_loss)
                    vali_loss = self.vali(vali_data, vali_loader, criterion)
                    print('Validation Finished (Vali)')
                    test_loss = self.vali(test_data, test_loader, criterion, label='test')
                    print('Validation Finished (Test)')
                    
                    print(model_optim)

                    print("Test Steps: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        self.steps, train_steps, tl, vali_loss, test_loss))
                    early_stopping(vali_loss, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        self.early_stop = True

            # Test on every epochs
            if self.test_every == 0:
                print("Test Steps: {} cost time: {}".format(self.test_every, time.time() - epoch_time))
                self.writer.add_scalar(f'LR/LearningRate', float(scheduler.get_last_lr()[0]), self.vali_times)
                tl = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                print('Validation Finished (Vali)')
                test_loss = self.vali(test_data, test_loader, criterion, label='test')
                print('Validation Finished (Test)')
                
                print(model_optim)

                print("Test Steps: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    self.steps, train_steps, tl, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    self.early_stop = True
                    
            model_optim.zero_grad()
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            if self.early_stop:
                break
                
            # Refresh train_loader
            train_data, train_loader = self._get_data(flag='train')

        best_model_path = path + '/' + 'checkpoint.pth'
        state_dict = torch.load(best_model_path)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict, strict=False)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            state_dict = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
                
            self.model.load_state_dict(new_state_dict, strict=False)

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                batch_x = data[0].float().to(self.device)
                batch_y = data[1].float().to(self.device)
                batch_x_mark = data[2].float().to(self.device)
                batch_y_mark = data[3].float().to(self.device)
                
                pred_len = self.args.pred_len # batch_y.shape[1] - batch_x.shape[1]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if 'CATS' in self.args.model:
                            outputs, masks, x_predictor_snapshot = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_mask=True)
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'CATS' in self.args.model:
                        outputs, masks, x_predictor_snapshot = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_mask=True)
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -self.args.number_of_targets
                if self.args.features == 'MS':
                    f_dim = -1
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)
                outputs = outputs.float().detach().cpu().numpy()
                batch_y = batch_y.float().detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.float().detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.float().detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        print(preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.close()

        np.save(folder_path + 'pred.npy', preds)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(pred_loader):
                batch_x = data[0].float().to(self.device)
                batch_y = data[1].float()
                batch_x_mark = data[2].float().to(self.device)
                batch_y_mark = data[3].float().to(self.device)
                pred_len = self.args.pred_len # batch_y.shape[1] - batch_x.shape[1]

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if 'CATS' in self.args.model:
                            outputs, masks, x_predictor_snapshot = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_mask=True)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'CATS' in self.args.model:
                        outputs, masks, x_predictor_snapshot = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_mask=True)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy(force=True)  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
    
        
