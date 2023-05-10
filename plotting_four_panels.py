import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
# mpl.use('Agg')         # USE IF WANTING TO SAVE THE FIGURE. TO SIMPLY DISPLAY FIGURE, COMMENT THIS OUT
mpl.rcParams['figure.dpi'] = 125

csv_file_name = '2_layers_1000_epochs_lr7_2_params/2_layers_1000_epochs_lr7_2_params_dataframe.csv'
temp_df = pd.read_csv(csv_file_name, index_col=0)
col_names = temp_df.columns.values.tolist()
cosmo_test_labels = ['00', '05', '07', '12', '18', 'fid']



def plot_loss(df, model_name, save_plot=False):
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    fig.suptitle('CNN Mean Loss vs. Epoch - 2 Params 6 Layers for 1000 Epochs (lr = e-6)', fontsize=14)
    fig.tight_layout(pad=4)

    df.plot(ax=axs, kind='line', x='Epoch', y='Loss', ylabel='Mean Loss', style=['b-'], ylim=[0, 1], legend=False)

    plt.show()

    if save_plot:
        plt.savefig(model_name + '_loss_plot', dpi=125)

    return fig


def plot_panels_acc(df, model_name, save_plot=False):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('CNN Mean Accuracy vs. Epoch - 6 Layers for 3000 Epochs (lr = e-7)', fontsize=14)
    fig.tight_layout(pad=4)

    om_list_acc = ['cosmo_' + cosmo_test_labels[z] + '_omegam_mean_accuracy' for z in range(6)]
    s8_list_acc = ['cosmo_' + cosmo_test_labels[z] + '_s8_mean_accuracy' for z in range(6)]
    h_list_acc = ['cosmo_' + cosmo_test_labels[z] + '_h_mean_accuracy' for z in range(6)]
    w0_list_acc = ['cosmo_' + cosmo_test_labels[z] + '_w0_mean_accuracy' for z in range(6)]

    df.plot(ax=axs[0, 0], title='$\Omega_{m}$', kind='line', x='Epoch', y=om_list_acc, ylabel='Accuracy', legend=False,
            ylim=[-0.2, 0.2], style=['b-', 'g-', 'r-', 'c-', 'm-', 'y-'])
    df.plot(ax=axs[0, 1], title='$S_{8}$', kind='line', x='Epoch', y=s8_list_acc, ylabel='Accuracy', legend=False,
            ylim=[-0.2, 0.2], style=['b-', 'g-', 'r-', 'c-', 'm-', 'y-'])
    df.plot(ax=axs[1, 0], title='$h$', kind='line', x='Epoch', y=h_list_acc, ylabel='Accuracy', legend=False,
            ylim=[-0.2, 0.2], style=['b-', 'g-', 'r-', 'c-', 'm-', 'y-'])
    df.plot(ax=axs[1, 1], title='$w_{0}$', kind='line', x='Epoch', y=w0_list_acc, ylabel='Accuracy',
            label=cosmo_test_labels, legend=False, ylim=[-0.2, 0.2], style=['b-', 'g-', 'r-', 'c-', 'm-', 'y-'])

    handles, labels = axs[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.show()

    if save_plot:
        plt.savefig(model_name + '_accuracy_plot', dpi=125)

    return fig


def plot_panels_true_and_pred_vals(df, model_name, save_plot=False):  # plots the true and pred values against epoch #

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('CNN True Value and Mean Predicted Value vs. Epoch - 6 Layers for 3000 Epochs (lr = e-7)'
                 '\nTrue Values -> Solid Lines. Predicted Values -> Dotted Lines', fontsize=14)
    fig.tight_layout(pad=4)
    labels = ['00', '05', '07', '12', '18', 'fid']

    om_list_mean_pred_val = ['cosmo_' + cosmo_test_labels[z] + '_omegam_mean_predicted_value' for z in range(6)]
    om_list_true_val = ['cosmo_' + cosmo_test_labels[z] + '_omegam_true_value' for z in range(6)]
    s8_list_mean_pred_val = ['cosmo_' + cosmo_test_labels[z] + '_s8_mean_predicted_value' for z in range(6)]
    s8_list_true_val = ['cosmo_' + cosmo_test_labels[z] + '_s8_true_value' for z in range(6)]
    h_list_mean_pred_val = ['cosmo_' + cosmo_test_labels[z] + '_h_mean_predicted_value' for z in range(6)]
    h_list_true_val = ['cosmo_' + cosmo_test_labels[z] + '_h_true_value' for z in range(6)]
    w0_list_mean_pred_val = ['cosmo_' + cosmo_test_labels[z] + '_w0_mean_predicted_value' for z in range(6)]
    w0_list_true_val = ['cosmo_' + cosmo_test_labels[z] + '_w0_true_value' for z in range(6)]

    df.plot(ax=axs[0, 0], title='$\Omega_{m}$', kind='line', x='Epoch', y=om_list_mean_pred_val,
            ylabel='True/Predicted Value', style=['b:', 'g:', 'r:', 'c:', 'm:', 'y:'], legend=False)
    df.plot(ax=axs[0, 0], title='$\Omega_{m}$', kind='line', x='Epoch', y=om_list_true_val,
            ylabel='True/Predicted Value', style=['b-', 'g-', 'r-', 'c-', 'm-', 'y-'], legend=False, ylim=[0.25, 0.5])
    df.plot(ax=axs[0, 1], title='$S_{8}$', kind='line', x='Epoch', y=s8_list_mean_pred_val,
            ylabel='True/Predicted Value', style=['b:', 'g:', 'r:', 'c:', 'm:', 'y:'], legend=False)
    df.plot(ax=axs[0, 1], title='$S_{8}$', kind='line', x='Epoch', y=s8_list_true_val, ylabel='True/Predicted Value',
            style=['b-', 'g-', 'r-', 'c-', 'm-', 'y-'], legend=False, ylim=[0.6, 0.85])
    df.plot(ax=axs[1, 0], title='$h$', kind='line', x='Epoch', y=h_list_mean_pred_val, ylabel='True/Predicted Value',
            style=['b:', 'g:', 'r:', 'c:', 'm:', 'y:'], legend=False)
    df.plot(ax=axs[1, 0], title='$h$', kind='line', x='Epoch', y=h_list_true_val, ylabel='True/Predicted Value',
            style=['b-', 'g-', 'r-', 'c-', 'm-', 'y-'], legend=False, ylim=[0.6, 0.8])
    df.plot(ax=axs[1, 1], title='$w_{0}$', kind='line', x='Epoch', y=w0_list_true_val, ylabel='True/Predicted Value',
            style=['b-', 'g-', 'r-', 'c-', 'm-', 'y-'], legend=False, ylim=[-1.85, -0.9])
    df.plot(ax=axs[1, 1], title='$w_{0}$', kind='line', x='Epoch', y=w0_list_mean_pred_val,
            ylabel='True/Predicted Value', style=['b:', 'g:', 'r:', 'c:', 'm:', 'y:'], legend=False)

    handles, labels2 = axs[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.show()

    if save_plot:
        plt.savefig(model_name + '_true_and_predicted_values_plot', dpi=125)

    return fig


plot_loss(temp_df, 'loss_test_6_layers_500_epochs_lr6_2_params')
acc_fig = plot_panels_acc(temp_df, '2_params_6_layers_5_epochs_lr6')
true_pred_fig = plot_panels_true_and_pred_vals(temp_df, '2_params_6_layers_5_epochs_lr6')


# plot_panels_acc_vs_pred_val(temp_df, '6_layers_1000_epochs_lr6')