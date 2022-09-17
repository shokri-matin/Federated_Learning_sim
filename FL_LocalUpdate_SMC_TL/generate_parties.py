# imports
import gzip
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle
import data_holder_party_class
from get_train_test_data import load_data_for_transfer_model


def generate_parties(num_data_holder_parties, tf_seed, num_local_updates, scenario, iteration):
    """ This function generates data holder parties and return a list of generated parties"""

    # load data

    x_train_transfer, x_val_transfer, x_test_transfer, y_train_transfer, y_val_transfer, y_test_transfer = load_data_for_transfer_model()
    
    num_train_records = x_train_transfer.shape[0]

    # data information
    train_data_record_indices = range(0, num_train_records)
    train_data_record_indices_shuffled = shuffle(train_data_record_indices, random_state=0)

    # indices of training data samples for each data holder party
    chunk_indices = np.array_split(train_data_record_indices_shuffled, num_data_holder_parties)

    parties = []
    for i in range(num_data_holder_parties):
        parties.append(data_holder_party_class.Party(data=x_train_transfer[chunk_indices[i]],
                                                     data_labels=y_train_transfer[chunk_indices[i]],
                                                     tf_seed=tf_seed,
                                                     num_local_updates=num_local_updates,
                                                     num_parties=num_data_holder_parties,
                                                     party_id=i,
                                                     scenario=scenario,
                                                     iteration=iteration))

    print("A list of parties have been created.\n\
    Data samples randomly assigned to each party.\n\
    The number of samples for each party is almost the same.")

    return parties
