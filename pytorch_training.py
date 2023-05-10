import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Holds functions: activation, convolutional, pooling, dropout, etc.
import numpy as np
import random
import pandas as pd

# Directories with source data
folder_path = '/home/bengib/NeuralNets/Learning_From_cosmoSLICS/QuickReadData/' \
              'MockKV450_DataKappa_SplitFid_zBins5_NoiseNone_AugFalse_Res128/'
train_file_name = 'Train_Data.npy'
train_params_file_name = 'Train_Cosmol_numPCosmol4.npy'
test_file_name = 'Test_Data.npy'
test_params_file_name = 'Test_Cosmol_numPCosmol4.npy'

# Train and test data as numpy array
train_dat = np.load(folder_path + train_file_name)
test_dat = np.load(folder_path + test_file_name)

# Parameters (cosmological constants) corresponding to each map in the train and test data
train_params = torch.from_numpy(np.load(folder_path + train_params_file_name)).double()
test_params = np.load(folder_path + test_params_file_name)

# Labels for the cosmologies used in the test set
cosmo_test_labels = ['00', '05', '07', '12', '18', 'fid']

# Normalizing the train and test data
mean = np.mean(train_dat)
sigma = np.std(train_dat)

norm_train = ((train_dat - mean) / sigma)
norm_test = ((test_dat - mean) / sigma)

train_data_norm = torch.from_numpy(norm_train).double()
test_data_norm = torch.from_numpy(norm_test).double()


class TestNet(nn.Module):
    def __init__(self, batch_size, num_layers, lr, pool_idx_list=[1, 3]):
        super().__init__()
        self.lr = lr
        self.num_channels = 5
        self.batch_size = batch_size
        self.stride = 1
        self.padding = 1                       # changed this to 1 from 0; this shouldn't change the size. still 128x128
        self.filter_size = 3
        self.input_size = 1
        self.input_size = 128
        self.nclayers = num_layers
        self.pool_idx_list = pool_idx_list  # list of indices after which to pool.
                                            # (i.e. for 4 convolutions, [1,3] would pool after 2nd and 4th convs)
        if self.nclayers == 2:
            self.pool_idx_list = [1]
        self.pools_count = len(self.pool_idx_list)  # number of times pooling; used to determine activation_size for correct output shape
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.num_channels, 5, self.filter_size, padding=self.padding, stride=self.stride) for i in range(self.nclayers)])
        self.pool = nn.MaxPool2d(2, 2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.out_size_const = 0.5
        if ((self.input_size - self.filter_size + 2 * self.padding) / self.stride).is_integer():
            self.out_size_const = 1

        # Activation Size = (Variable Constant + (Input Size - Filter Size + 2*Padding)) / Stride
        # 0.5^pools_count reduces dimensionality change from pooling
        self.activation_size = 0.5 ** self.pools_count * (
                    self.out_size_const + (self.input_size - self.filter_size + 2 * self.padding) / self.stride)

        if (5 * self.activation_size ** 2).is_integer():
            self.fc1 = nn.Linear(int(5 * self.activation_size ** 2), 4)  # of output params (4) for the number of cosmos
        else:
            # yield error if shape is not composed of integers (e.g. too many poolings)
            print("ERROR: FULLY CONNECTED LAYER SHAPE DOES NOT CONTAIN INTEGER")

    def forward(self, x):

        for i, layer in enumerate(self.convs):
            x = F.relu(layer(x))
            if i in self.pool_idx_list:
                x = self.pool(F.relu(layer(x)))
        x = torch.flatten(x, 1)              # flatten all dimensions except batch -> (i.e. (4, 5, 32, 32) to (4, 5120))
        x = self.fc1(x)

        return x


def testing(model_name, batch_size, num_layers, lr):
    test_net = TestNet(batch_size, num_layers, lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_net.load_state_dict(torch.load(model_name, map_location=device))
    test_net.to(device).float()

    idx = np.arange(0, test_data_norm.shape[0])
    idx = np.reshape(idx, (int(len(idx) / batch_size), batch_size))
    test_predictions = np.zeros([test_data_norm.shape[0], test_params.shape[1]])
    iteration = 0

    for i in range(idx.shape[0]):
        iteration = iteration + 1
        inputs = test_data_norm[idx[i]].to(device)
        outputs = test_net(inputs.float())
        test_predictions[i * batch_size:(i + 1) * batch_size, :] = outputs.detach().cpu().numpy()  # storing output preds

    unique_arrays, indices, counts = np.unique(test_params, axis=0, return_index=True, return_counts=True)

    init_range = 0
    num_cosmos = 4
    indices_list_sorted = np.sort(np.append(indices, len(test_params)))[1:]
    true_cosmologies = np.array([test_params[x] for x in np.sort(indices)])

    means = np.zeros_like(unique_arrays)
    stdevs = np.zeros_like(unique_arrays)
    mean_accs = np.zeros_like(unique_arrays)

    for i, x in enumerate(indices_list_sorted):
        cosmo_pred_array = np.empty([counts[i], num_cosmos])

        for y in range(init_range, x):
            cosmo_pred_array[y - init_range] = test_predictions[y]

        mean_vals = np.mean(cosmo_pred_array, axis=0)
        stdev_vals = np.std(cosmo_pred_array, axis=0) / np.sqrt(
            len(cosmo_pred_array[0]))                      # divide stdev by sqrt(96) because more data means less error

        # ACCURACY CALCULATION IS INCORRECT!!!!! IGNORE THIS COMING LINE
        mean_acc_vals = np.divide(mean_vals - np.array(test_params[i]), np.abs(test_params[i])) # (mean pred-truth)/|truth|
        means[i] = mean_vals                # mean of the predictions
        stdevs[i] = stdev_vals              # standard deviation of the predictions
        mean_accs[i] = mean_acc_vals        # accuracy of the predictions based of mean of predictions and true values
        init_range = x

    print('Finished Testing')

    return means, stdevs, mean_accs, true_cosmologies, test_predictions


def train_with_test_with_dataframe(model_name, batch_size, num_layers, num_epochs, lr):
    # making a file that holds the accuracy results
    f = open(model_name + '_accuracy_file.txt', 'w')    # open or create the file
    f.truncate()                                        # deletes the file contents if it exists already
    f.write('Data As Text\n')

    test_net = TestNet(batch_size, num_layers, lr)

    n_maps = train_data_norm.shape[0]       # number of maps
    num_batches = int(n_maps / batch_size)  # number of batches
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_net.to(device).float()

    col_names = ['Epoch']

    for x in range(6):
        new_names_mean_accuracy = ['cosmo_' + cosmo_test_labels[x] + '_omegam_mean_accuracy',
                                   'cosmo_' + cosmo_test_labels[x] + '_s8_mean_accuracy',
                                   'cosmo_' + cosmo_test_labels[x] + '_h_mean_accuracy',
                                   'cosmo_' + cosmo_test_labels[x] + '_w0_mean_accuracy']
        col_names = col_names + new_names_mean_accuracy
    for x in range(6):
        new_names_stdev = ['cosmo_' + cosmo_test_labels[x] + '_omegam_ma_stdev',
                           'cosmo_' + cosmo_test_labels[x] + '_s8_ma_stdev',
                           'cosmo_' + cosmo_test_labels[x] + '_h_ma_stdev',
                           'cosmo_' + cosmo_test_labels[x] + '_w0_ma_stdev']
        col_names = col_names + new_names_stdev

    for x in range(6):
        new_names_mean_pred_val = ['cosmo_' + cosmo_test_labels[x] + '_omegam_mean_predicted_value',
                                   'cosmo_' + cosmo_test_labels[x] + '_s8_mean_predicted_value',
                                   'cosmo_' + cosmo_test_labels[x] + '_h_mean_predicted_value',
                                   'cosmo_' + cosmo_test_labels[x] + '_w0_mean_predicted_value']
        col_names = col_names + new_names_mean_pred_val

    for x in range(6):
        new_names_true_val = ['cosmo_' + cosmo_test_labels[x] + '_omegam_true_value',
                              'cosmo_' + cosmo_test_labels[x] + '_s8_true_value',
                              'cosmo_' + cosmo_test_labels[x] + '_h_true_value',
                              'cosmo_' + cosmo_test_labels[x] + '_w0_true_value']
        col_names = col_names + new_names_true_val

    col_names = col_names + ['Loss']

    all_data_array = np.zeros((round(num_epochs/2), 98))     # shape is (# of epochs/2 + 1, # of columns -> sd + mean + pred val + true val). the +1 is for 0th epoch

    index_of_all_data_array = 0
    for epoch in range(num_epochs):

        rand_idx = np.arange(0, train_data_norm.shape[0])
        np.random.seed(1)
        random.shuffle(rand_idx)
        rand_idx = np.reshape(rand_idx, (int(len(rand_idx) / batch_size), batch_size))

        all_loss = []

        for i in range(num_batches):
            inputs = train_data_norm[rand_idx[i]].to(device)   # data points for subsequent label values
            labels = train_params[rand_idx[i]].to(device)      # cosmology values
            test_net.optimizer.zero_grad()                     # zero the parameter gradients

            # forward + backward + optimize
            outputs = test_net(inputs.float())
            loss = test_net.criterion(outputs.float(), labels.float())
            loss.backward()
            test_net.optimizer.step()
            all_loss.append(loss.item())

        mean_loss = np.mean(np.array(all_loss))

        # testing during training every 2 epochs. Weights are saved every two epochs and m, sd, ma, tc, tp
        if epoch % 2 == 0:
            torch.save(test_net.state_dict(), model_name + '_epoch_' + str(epoch))
            m, sd, ma, tc, tp = testing(model_name + '_epoch_' + str(epoch), batch_size, num_layers, lr)
            f.write('\nEPOCH ' + str(epoch))
            f.write('\nMean\n')
            f.write(str(m))
            f.write('\nStandard Deviation\n')
            f.write(str(sd))
            f.write('\nMean Accuracies\n')
            f.write(str(ma))
            f.write('\nTrue Cosmologies\n')
            f.write(str(tc))
            f.write('\nTest Predictions\n')
            f.write(str(tp))
            f.write('\nTraining Mean Loss\n')
            f.write(str(mean_loss))

            # Add mean accuracy data to the all data array from index 1-24
            count_ma = 1
            for j1 in ma:
                for k1 in j1:
                    all_data_array[index_of_all_data_array][count_ma] = k1
                    count_ma = count_ma + 1

            # Add mean accuracy standard deviation data to the all data array from index 25-48
            count_sd = 25
            for j2 in sd:
                for k2 in j2:
                    all_data_array[index_of_all_data_array][count_sd] = k2
                    count_sd = count_sd + 1

            # Add mean predicted value data to the all data array from index 49-72
            count_pred = 49
            for j3 in m:
                for k3 in j3:
                    all_data_array[index_of_all_data_array][count_pred] = k3
                    count_pred = count_pred + 1

            # Add predicted value data to the all data array from index 73-96
            count_true = 73
            for j4 in tc:
                for k4 in range(4):
                    all_data_array[index_of_all_data_array][count_true] = j4[k4]
                    count_true = count_true + 1

            all_data_array[index_of_all_data_array][-1] = mean_loss                # Add loss to array
            all_data_array[index_of_all_data_array][0] = epoch                     # Add epoch number to array
            index_of_all_data_array = index_of_all_data_array + 1                  # Add next index value to array
    print('Finished Training')

    df = pd.DataFrame(all_data_array, columns=col_names)
    torch.save(test_net.state_dict(), model_name)
    f.close()

    df.to_csv(model_name + '_dataframe.csv')

    return test_net, df


# these are my testing values
net, test_df = train_with_test_with_dataframe("6_layers_3000_epochs_lr5", 4, 6, 3000, 1e-6)
