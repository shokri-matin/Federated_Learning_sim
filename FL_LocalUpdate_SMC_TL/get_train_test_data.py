import h5py
import numpy as np
from sklearn.model_selection import train_test_split

def load_data_for_transfer_model(path="test_train_data"):

    x_train_transfer = np.load(f"{path}/x_train_transfer.npy")
    y_train_transfer = np.load(f"{path}/y_train_transfer.npy")

    x_test_transfer = np.load(f"{path}/x_test_transfer.npy")
    y_test_transfer = np.load(f"{path}/y_test_transfer.npy")

    x_val_transfer = np.load(f"{path}/x_val_transfer.npy")
    y_val_transfer = np.load(f"{path}/y_val_transfer.npy")

    return x_train_transfer, x_val_transfer, x_test_transfer, y_train_transfer, y_val_transfer, y_test_transfer

def generate_train_test_data(path):

    window = 1024
    size = 2*window
    ds_rate = 1

    k = 0
    for i in range(23):
        rd1 = np.random.randint(0, 17, size=1)[0]
        rd2 = np.random.randint(0, 17, size=1)[0]
        j = i + 1

        mat = h5py.File(f"{path}/chb{j:02d}.mat".format(path))
        temp = np.transpose(np.array(mat.get('Signal_windows')))
        temp0 =  temp[:, 13 * window + 1:14 * window + 1:ds_rate]  # eglass eletrode
        temp1 = temp[:, 1 * window + 1:2 * window + 1:ds_rate]  # eglass eletrode
        temp2 = temp[:, rd1 * window + 1:(rd1+1) * window + 1:ds_rate]  # random electrode
        temp3 = temp[:, rd2 * window + 1:(rd2+1) * window + 1:ds_rate]  # random electrode

        Y = np.transpose(np.array(mat.get('Y')))
        temp_lab = Y
        temp_sig = np.concatenate((temp0+np.random.normal(0, 0.1*np.std(temp0), window),
                                  temp1+np.random.normal(0, 0.1*np.std(temp1), window), temp2, temp3), axis=1)
        if k == 0:
            all_sig = temp_sig.copy()
            all_label = temp_lab.copy()
        else:
            all_sig = np.concatenate((all_sig, temp_sig), axis=0)
            all_label = np.concatenate((all_label, temp_lab), axis=0)
        k += 1


    print("split the data into train, val, test")
    #split the data into train, val, test
    x = all_sig
    y = all_label

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    #other electrodes
    x_train_other=x_train[:,2*window:]
    x_val_other=x_val[:,2*window:]

    #eglass electrodes
    x_train_eglass=x_train[:,0:2*window]
    x_val_eglass=x_val[:,0:2*window]
    x_test_eglass=x_test[:,0:2*window]
    y_test_eglass=y_test
    
    x_train_other = np.reshape(x_train_other, (np.shape(x_train_other)[0], size, 1))
    x_val_other = np.reshape(x_val_other, (np.shape(x_val_other)[0], size, 1))

    x_train_eglass = np.reshape(x_train_eglass, (np.shape(x_train_eglass)[0], size, 1))
    x_val_eglass = np.reshape(x_val_eglass, (np.shape(x_val_eglass)[0], size, 1))
    x_test_eglass = np.reshape(x_test_eglass, (np.shape(x_test_eglass)[0], size, 1))

    #split the test set for eglass electrodes into train, val, test
    x = x_test_eglass
    y = y_test_eglass

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    x_train = np.reshape(x_train, (np.shape(x_train)[0], size, 1))
    x_val = np.reshape(x_val, (np.shape(x_val)[0], size, 1))
    x_test = np.reshape(x_test, (np.shape(x_test)[0], size, 1))

    np.save("test_train_data/x_train_transfer.npy", x_train)
    np.save("test_train_data/x_test_transfer.npy", x_test)
    np.save("test_train_data/x_val_transfer.npy", x_val)

    np.save("test_train_data/y_train_transfer.npy", y_train)
    np.save("test_train_data/y_test_transfer.npy", y_test)
    np.save("test_train_data/y_val_transfer.npy", y_val)