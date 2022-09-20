# imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf



class Server:

    def __init__(self, num_data_holder_parties, tf_seed):
        # self.model = None
        # self.define_model()
        self.global_model_parameters = None

        self.num_data_holder_parties = num_data_holder_parties

        tf.random.set_seed(tf_seed)

    def define_model(self):
        """ This function generates the NN model"""

        model_head = tf.keras.models.Sequential([
                            tf.keras.layers.InputLayer(input_shape=(1024)),
                            tf.keras.layers.Dense(8, kernel_regularizer=tf.keras.regularizers.L1(0.001)),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.Activation('relu'),
                            tf.keras.layers.Dropout(.3),
                            tf.keras.layers.Dense(2, activation=tf.nn.softmax)
                        ])

        self.model = model_head

        self.model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 'binary_crossentropy',
            metrics=['accuracy'],
        )

    def aggregate_model_parameters(self, model_parameters_dict):
        """ This function aggregates the received model parameters from data holder parties"""

        parties_model_parameters_list = list(model_parameters_dict.keys())

        self.global_model_parameters = model_parameters_dict[parties_model_parameters_list[0]]
        for i in range(1, self.num_data_holder_parties):
            temp = model_parameters_dict[parties_model_parameters_list[i]]
            for j in range(0, len(self.global_model_parameters)):
                self.global_model_parameters[j] += temp[j]

        for j in range(0, len(self.global_model_parameters)):
            self.global_model_parameters[j] /= self.num_data_holder_parties

    def interface_pipeline(self, model_parameters_dict):
        """ This can be called by interface """

        # receive model parameters from data holder parties
        # aggregate model parameters
        self.aggregate_model_parameters(model_parameters_dict)
        # share/send global_model_parameters to interface/data holder parties
        return self.global_model_parameters

