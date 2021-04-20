from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

class NetworkArchitecture:

    def __init__(self):

        self.createModelStructure()
        self.compileModel()
        self.plot_model()

    def createModelStructure(self, verbose=False):

        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(38, 20, 1)),
            MaxPool2D((2, 2), strides=2),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(80, activation='relu', kernel_initializer='he_uniform'),
            Dropout(0.2),
            Dense(36, activation='softmax')
        ])

        if verbose:
            print(self.model.summary())

    def compileModel(self):

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    def plot_model(self):

        img_file = 'outputs/model_arch.png'
        plot_model(
            self.model,
            to_file=img_file,
            show_shapes=True,
            show_layer_names=True
        )


