import tensorflow as tf
from keras import Input

img_height = 160
img_width = 160
num_channels = 3
IMG_SIZE = (160, 160)

def HumansModel():
    IMG_SHAPE = IMG_SIZE + (3,)

    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    base_model.summary()

    data_augmentation = tf.keras.Sequential([
        # tf.keras.layers.RandomFlip('horizontal'),
        # tf.keras.layers.RandomRotation(0.2),
        # tf.keras.layers.RandomBrightness((-0.2, 0.2))
    ])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)

    dense128Layer = tf.keras.layers.Dense(128, activation='relu')

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)

    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = dense128Layer(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.summary()

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])

    return model








    # old model, too simple
    # inputs = Input(shape=(img_height, img_width, num_channels))
    # model = tf.keras.Sequential([
    #     inputs,
    #     tf.keras.layers.Rescaling(1. / 255),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
    #
    # return model