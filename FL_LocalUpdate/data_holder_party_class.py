# imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf


class Party:

    def __init__(self, data, data_labels, tf_seed, num_local_updates):
        self.data = data
        self.data_labels = data_labels
        # Instantiate a loss function.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.grads = None
        self.num_local_updates = num_local_updates
        self.tf_seed = tf_seed
        self.model = self.define_model()


    def define_model(self):
        """ This function generates the NN model"""

        # def my_init(shape, dtype=None):
        #     return tf.keras.backend.random_normal(shape, dtype=dtype, seed=self.tf_seed)
        # initializer = tf.keras.initializers.GlorotUniform(seed=self.tf_seed)
        # initializer = tf.keras.initializers.Zeros()
        model = keras.Sequential([
            layers.InputLayer(input_shape=[28, 28]),

            # Head
            layers.BatchNormalization(renorm=True),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),  # , kernel_initializer=initializer or my_init (for no simulation)
            layers.Dense(10),  # , kernel_initializer=initializer or my_init (for no simulation)

        ])

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
            # calculate gradients
            self.calculate_gradients(self.data, self.data_labels)
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
        # share locally updated model parameters
        return self.locally_update_model()




