# import server_party_class
# import generate_parties
# import os
# from pathlib import Path
# import gzip
# import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
# from matplotlib import pyplot as plt
# import tensorflow as tf

from get_train_test_data import generate_train_test_data

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# generate test and train data and save those in test_train_data folder
generate_train_test_data(path="dataset")

# # load test data
# data_dir = Path('C:/Amin/Workspace/Data/fashion mnist')

# # load test data
# f = gzip.open(data_dir/'t10k-images-idx3-ubyte.gz','r')
# image_size = 28
# num_images = 10000


# f.read(16)
# buf = f.read(image_size * image_size * num_images)
# test_data_raw = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
# test_data_raw = test_data_raw.reshape(num_images, image_size, image_size, 1)

# # image = np.asarray(test_data[2]).squeeze()
# # plt.imshow(image, cmap="Greys")
# # plt.show()

# # labels
# f = gzip.open(data_dir/'t10k-labels-idx1-ubyte.gz','r')
# f.read(8)

# buf = f.read(num_images)
# test_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
# # print(labels)

# model_base = keras.Sequential([
#             # base
#             layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform',
#                           input_shape=(28, 28, 1)),
#             layers.MaxPooling2D((2, 2)),
#         ])
# checkpoint_path = "model_base/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model_base.load_weights(latest)

# test_data = model_base.predict(test_data_raw)
# # print(np.shape(test_data))


# # define the model only for evaluation
# model_head = keras.Sequential([
#     layers.InputLayer(input_shape=[14, 14, 32]),
#     # Head
#     layers.Flatten(),
#     layers.Dense(100, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

# model = model_head

# model.compile(
#     optimizer='adam',
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 'binary_crossentropy',
#     metrics=['accuracy'],
# )


# tf_seed = 0
# num_data_holder_parties = 5
# num_local_updates = 3
# scenario = 2

# # generate/instantiate parties
# data_holder_parties_all = generate_parties.generate_parties(num_data_holder_parties=num_data_holder_parties,
#                                                             tf_seed=tf_seed,
#                                                             num_local_updates=num_local_updates,
#                                                             scenario=scenario)
# server_party = server_party_class.Server(num_data_holder_parties=num_data_holder_parties,
#                                          tf_seed=tf_seed)


# # repeat training process (as the interface)
# global_model_parameters = None
# num_epoch = 10  # 100
# test_loss, test_acc = np.zeros(num_epoch), np.zeros(num_epoch)
# for epoch in range(num_epoch):
#     print("iteration", epoch)
#     try:
#         model_parameters_dict
#         model_parameters_dict.clear()
#     except NameError:
#         pass
#     model_parameters_dict = {}
#     for data_holder_i in range(0, num_data_holder_parties):
#         if global_model_parameters is None:
#             model_parameters_dict[data_holder_i] = data_holder_parties_all[data_holder_i].interface_pipeline()
#         else:
#             model_parameters_dict[data_holder_i] = \
#                 data_holder_parties_all[data_holder_i].interface_pipeline(global_model_parameters)
#     global_model_parameters = server_party.interface_pipeline(model_parameters_dict)


#     # for evaluation purpose only
#     model.set_weights(global_model_parameters)
#     test_loss[epoch], test_acc[epoch] = model.evaluate(test_data, test_labels, verbose=2)

# # plot evaluation results
# fig,  axs = plt.subplots(1, 2)
# axs[0].set_title("test_loss")
# axs[0].plot(range(1, num_epoch+1), test_loss)
# axs[0].set(xlabel='epoch', ylabel='test_loss')

# axs[1].set_title("test_acc")
# axs[1].plot(range(1, num_epoch+1), test_acc)
# axs[1].set(xlabel='epoch', ylabel='test_acc')
# plt.show()



