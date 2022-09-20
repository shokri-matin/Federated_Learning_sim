import tensorflow as tf

def get_all_model(head_size = 8):

    num_class = 2
    dropout_rate = .3

    model_base = tf.keras.models.Sequential([
                tf.keras.layers.Conv1D(512, kernel_size=2, strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv1D(128, kernel_size=2, strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv1D(4, kernel_size=2, strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Flatten(),
                ])

    model_head = tf.keras.models.Sequential([
                        tf.keras.layers.InputLayer(input_shape=(1024)),
                        tf.keras.layers.Dense(head_size, kernel_regularizer=tf.keras.regularizers.L1(0.001)),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation('relu'),
                        tf.keras.layers.Dropout(dropout_rate),
                        tf.keras.layers.Dense(num_class, activation=tf.nn.softmax)
                    ])


    return model_base, model_head

