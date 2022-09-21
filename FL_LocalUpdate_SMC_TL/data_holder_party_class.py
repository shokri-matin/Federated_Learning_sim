# imports
from email import iterators
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf
import SMC_functions
import os
from sklearn.utils import shuffle
import numpy as np

class Party:

    def __init__(self, data, data_labels, tf_seed, num_local_updates, num_parties, party_id, scenario, iteration):
        self.data_raw = data
        self.data = None
        self.iteration = iteration
        self.predict_with_model_base()
        self.data_labels = data_labels
        # Instantiate a loss function.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.grads = None
        self.num_local_updates = num_local_updates
        self.tf_seed = tf_seed
        self.model = self.define_model()
        
        # self.SMC_tools = SMC_functions.SMCtools(num_parties=num_parties, party_id=party_id,
        #                                         num_participating_parties=num_parties,
        #                                         secure_aggregation_parameter_k=num_parties - 1,
        #                                         scenario=scenario)

    def predict_with_model_base(self):

        model_base = keras.models.load_model("model_base/{}/model_base.h5".format(self.iteration))
        self.data = model_base.predict(self.data_raw)

    def define_model(self):
        """ This function generates the NN model"""

        model_head = tf.keras.models.Sequential([tf.keras.layers.Dense(8, kernel_regularizer=tf.keras.regularizers.L1(0.001)),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation('relu'),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(2, activation=tf.nn.softmax)])

        model = model_head

        model.compile(
            optimizer='adam',
            loss=self.loss_fn,  # 'binary_crossentropy', categorical_crossentropy
            metrics=['accuracy'],
        )
        return model

    def calculate_gradients(self, x, y):
        """ This function calculates gradients for one round of feedforward and back propagation """

        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
        self.grads = tape.gradient(loss_value, self.model.trainable_weights)

    def locally_update_model(self):
        """ This function updates the model one to several times based on local data
         and returns the updated parameters """

        for i in range(0, self.num_local_updates):

            train_data_record_indices = range(0, self.data.shape[0])
            train_data_record_indices_shuffled = shuffle(train_data_record_indices, random_state=0)

            batch_size = 16
            num_batches = int(self.data.shape[0]/batch_size)
            chunk_indices = np.array_split(train_data_record_indices_shuffled, num_batches)

            for j in range(num_batches):
                # calculate gradients
                self.calculate_gradients(self.data[chunk_indices[j]], self.data_labels[chunk_indices[j]])
                # update model based on calculated grads
                self.model.optimizer.apply_gradients(zip(self.grads, self.model.trainable_weights))

        # return locally updated model parameters
        return self.model.get_weights()

    def interface_pipeline(self, global_model_parameters=None):
        """ This can be called by server/interface """

        if global_model_parameters is not None:
            #  receive global model parameters
            # update local model by global_model_parameters
            self.model.set_weights(global_model_parameters)
        # locally update the model
        self.locally_update_model()
        model_parameters = self.locally_update_model()
        
        # mask updated model parameters
        # masked_model_parameters = self.SMC_tools.mask(model_parameters)
        # share masked updated model parameters
        masked_model_parameters = model_parameters
        return masked_model_parameters




