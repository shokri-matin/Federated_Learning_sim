import sys
import server_party_class
import os
import numpy as np
from tensorflow import keras
import generate_parties

from os import path

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from get_train_test_data import generate_train_test_data, load_data_for_transfer_model
from model import get_all_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def run(argv):

    np.random.seed(seed=42)

    iteration = int(argv[1]) # 100
    
    num_data_holder_parties = int(argv[2]) # 5
    num_local_updates = int(argv[3]) # 3
    scenario = int(argv[4]) # 2

    tf_seed = 0

    num_epoch = int(argv[5]) # 100

    test_acc, test_pre, test_re, test_f1 = np.zeros([iteration, num_epoch]),\
         np.zeros([iteration, num_epoch]),\
         np.zeros([iteration, num_epoch]),\
         np.zeros([iteration, num_epoch])

    for it in range(iteration):
        print("iteration: {}".format(it))
        # generate test and train data and save those in test_train_data folder
        print("1-generate test and train data and save those in test_train_data folder")
        generate_train_test_data(path="dataset")

        # train the all model
        print("3-load all model")
        model_base, model_head = get_all_model(head_size=8)

        model_head.compile(
                optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
            )

        model_base = keras.models.load_model("model_base/{}/model_base.h5".format(it))

        print("6-preparing model for transfer learning")

        # generate/instantiate parties
        print("7-generate/instantiate parties")
        data_holder_parties_all = generate_parties.generate_parties(num_data_holder_parties=num_data_holder_parties,
                                                                    tf_seed=tf_seed,
                                                                    num_local_updates=num_local_updates,
                                                                     scenario=scenario,
                                                                    iteration=it)

        print("8-generate/instantiate mediator")
        server_party = server_party_class.Server(num_data_holder_parties=num_data_holder_parties,
                                                tf_seed=tf_seed)

        # repeat training process (as the interface)
        global_model_parameters = None

        x_train_transfer, x_val_transfer, x_test_transfer, y_train_transfer, y_val_transfer, y_test_transfer = load_data_for_transfer_model()

        x_test_transfer_predicted = model_base.predict(x_test_transfer)

        for epoch in range(num_epoch):
            print("epoch", epoch)
            try:
                model_parameters_dict
                model_parameters_dict.clear()
            except NameError:
                pass
            model_parameters_dict = {}
            for data_holder_i in range(0, num_data_holder_parties):
                if global_model_parameters is None:
                    model_parameters_dict[data_holder_i] = data_holder_parties_all[data_holder_i].interface_pipeline()
                else:
                    model_parameters_dict[data_holder_i] = \
                        data_holder_parties_all[data_holder_i].interface_pipeline(global_model_parameters)
            global_model_parameters = server_party.interface_pipeline(model_parameters_dict)

            model_head.set_weights(global_model_parameters)

            y_pred = model_head.predict(x_test_transfer_predicted)
            y_pred = np.argmax(y_pred, axis=1)

            test_acc[it, epoch] = accuracy_score(y_test_transfer, y_pred)
            test_pre[it, epoch] = precision_score(y_test_transfer, y_pred)
            test_re[it, epoch] = recall_score(y_test_transfer, y_pred)
            test_f1[it, epoch] = f1_score(y_test_transfer, y_pred)

            print('accuracy_score :', accuracy_score(y_test_transfer, y_pred))
            print('recall_score: ', recall_score(y_test_transfer, y_pred))
            print('precision_score: ', precision_score(y_test_transfer, y_pred))
            print('f1_score: ', f1_score(y_test_transfer, y_pred))
    
    mean_test_acc = np.mean(test_acc, axis=0)
    mean_test_pre = np.mean(test_pre, axis=0)
    mean_test_re = np.mean(test_re, axis=0)
    mean_test_f1 = np.mean(test_f1, axis=0)

    print("Final Accuracy: ", mean_test_acc)
    print("Final Precision: ", mean_test_pre)
    print("Final Recall: ", mean_test_re)
    print("Final F1: ", mean_test_f1)

    np.save("Results/mean_test_acc_{}".format(iteration), mean_test_acc)
    np.save("Results/mean_test_pre_{}".format(iteration), mean_test_pre)
    np.save("Results/mean_test_re_{}".format(iteration), mean_test_re)
    np.save("Results/mean_test_f1_{}".format(iteration), mean_test_f1)

if __name__  == "__main__":
    run(sys.argv)

