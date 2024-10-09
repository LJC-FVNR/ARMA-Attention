import numpy as np
from IPython.display import display, Markdown, Latex, HTML
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns

def vis_channel_forecasting(L_I, X, X_adj, X_true, masks, col_names=None):
    """
    Modified function to plot the time series data for visualization with mask highlighting
    that fills the entire vertical range between vmin and vmax.

    Parameters:
    - L_I (int): Length of the input segment.
    - X (torch.Tensor): Tensor containing the preprocessed input and output of the time series model.
                        Shape: (L_I + L_P, C)
    - X_adj (torch.Tensor): Tensor containing the adjusted predictions for a subset of the series.
                            Shape: (L_P, C_s)
    - X_true (torch.Tensor): Tensor containing the ground truth data.
                             Shape: (L_I + L_P, C_s)
    - masks (torch.Tensor): Tensor containing masking information. Shape: (L_I, C)
    """
    L_P = X.shape[0] - L_I  # Predicted length
    C = X.shape[1]  # Number of channels (time series)
    C_s = X_adj.shape[1]  # Number of adjusted series

    fig, axs = plt.subplots(C, 1, figsize=(10, 2 * C), sharex=True, dpi=72)
    
    X_adj = torch.cat([X[[L_I-1], -C_s:], X_adj], dim=0)
    
    X = X.detach().cpu().numpy()
    X_adj = X_adj.detach().cpu().numpy()
    X_true = X_true.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    
    # Check if there is only one time series (C == 1)
    if C == 1:
        axs = [axs]

    for i in range(C):
        # Plot input and output for each time series
        axs[i].plot(range(L_I), X[:L_I, i], label='Input', color='blue', alpha=0.75)
        axs[i].plot(range(L_I-1, L_I + L_P), X[(L_I-1):, i], label='Output', color='orange', alpha=0.75)

        # If this series has adjusted predictions and ground truth
        if i >= C - C_s:
            axs[i].plot(range(L_I-1, L_I + L_P), X_adj[:, i - (C - C_s)], label='Forecasting', color='green', alpha=0.75)
            axs[i].plot(range(0, L_I + L_P), X_true[:, i - (C - C_s)], label='True', color='red', alpha=0.75)

        # Draw a vertical line to show the separation between input and output
        axs[i].axvline(x=L_I-1, color='gray', linestyle='--')

        # Find the masking point and fill the area
        t_i = np.where(masks[:, i] == 1)[0][0] if 1 in masks[:, i] else L_I
        ymin, ymax = axs[i].get_ylim()
        axs[i].fill_between(range(t_i, L_I), ymin, ymax, color='green', alpha=0.1)

        # Adding legend and labels
        axs[i].legend()
        if col_names is None or C-i > len(col_names):
            axs[i].set_ylabel(f"Series (-{C-i})")
        else:
            axs[i].set_ylabel(f"{col_names[-(C-i)]} (-{C-i})")

    axs[-1].set_xlabel("Time")
    plt.tight_layout()
    return fig

def plot_aligned_heatmap(x):  
    fig = plt.figure(figsize=(12, 12), dpi=120)

    p1 = plt.subplot2grid((20,21),(0,0), rowspan=16, colspan=3)
    p2 = plt.subplot2grid((20,21),(0,3), rowspan=16, colspan=16)
    p3 = plt.subplot2grid((20,21),(16,3), rowspan=3, colspan=16)
    p4 = plt.subplot2grid((20,21),(0,19), rowspan=16, colspan=2)

    
    im = p2.imshow(x, cmap=sns.cubehelix_palette(as_cmap=True, reverse=True), aspect="auto")
    fig.colorbar(im, cax=p4)
    plt.subplots_adjust(wspace=100, hspace=100)
    p1.plot(x.mean(axis=1), range(x.shape[0]), color="#4B4453")
    p1.set_ylim(0, x.shape[0]-1)
    p1.invert_yaxis()
    p3.plot(range(x.shape[1]), x.mean(axis=0), color="#4B4453")
    p3.set_xlim(0, x.shape[1]-1)

    p1.yaxis.set_major_locator(MaxNLocator(integer=True))
    p2.xaxis.set_major_locator(MaxNLocator(integer=True))
    p2.yaxis.set_major_locator(MaxNLocator(integer=True))
    p3.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig

def mts_visualize(pred, true, split_step=720, title='Long-term Time Series Forecasting', dpi=72, col_names=None):
    groups = range(true.shape[-1])
    C = true.shape[-1]
    i = 1
    # plot each column
    f = plt.figure(figsize=(10, 2.1*len(groups)), dpi=dpi)
    f.suptitle(title, y=0.9)
    index = 0
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(true[:, group], alpha=0.75, label='True')
        if type(pred) is list:
            for index, p in enumerate(pred):
                plt.plot(list(range(split_step, true.shape[0])), p[:, group], alpha=0.5, label=f'Pred_{index}')
        else:
            plt.plot(list(range(split_step, true.shape[0])), pred[:, group], alpha=0.75, label='Pred')
        #plt.title(f'S{i}', y=1, loc='right')
        if col_names is None or C-index > len(col_names):
            plt.title(f"Series (-{C-index})", y=1, loc='right')
        else:
            plt.title(f"{col_names[-(C-index)]} (-{C-index})", y=1, loc='right')
        index += 1
        plt.legend(loc='lower left')
        plt.axvline(x=split_step, linewidth=1, color='Purple')
        i += 1
    return f

def mts_visualize_horizontal(pred, true, split_step=720, title='Long-term Time Series Forecasting', dpi=72, width=10, col_names=None):
    groups = range(true.shape[-1])
    C = true.shape[-1]
    i = 1
    # plot each column
    f = plt.figure(figsize=(width, 2.1*len(groups)), dpi=dpi)
    f.suptitle(title, y=0.9)
    index = 0
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(true[:, group], alpha=0.75, label='True')
        if type(pred) is list:
            for index, p in enumerate(pred):
                plt.plot(list(range(split_step, true.shape[0])), p[:, group], alpha=0.75, label=f'Pred_{index}', linestyle=':')
        else:
            plt.plot(list(range(split_step, true.shape[0])), pred[:, group], alpha=0.75, label='Pred')
        #plt.title(f'S{i}', y=1, loc='right')
        if col_names is None or C-index > len(col_names):
            plt.title(f"Series (-{C-index})", y=1, loc='right')
        else:
            plt.title(f"{col_names[-(C-index)]} (-{C-index})", y=1, loc='right')
        index += 1
        plt.legend(ncol=1000, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        plt.axvline(x=split_step, linewidth=1, color='Purple')
        i += 1
    return f