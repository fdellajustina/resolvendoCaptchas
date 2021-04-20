from tensorflow.keras.models import load_model
from tensorflow import reshape
from cv2 import imread, cvtColor, COLOR_BGR2GRAY
from string import ascii_lowercase
from numpy import argmax

class Predict:

    def __init__(self, imageTOpredict_str):

        self.classes_captchSimbols = ascii_lowercase + '0123456789'
        self.model = load_model('models/captcha.h5')
        self.imageTOpredict_str = imageTOpredict_str

    def set_preprocessing_image(self):

        self.image_predict = imread(f'samples/{self.imageTOpredict_str}')
        self.image_predict = cvtColor(self.image_predict, COLOR_BGR2GRAY)
        self.image_predict = self.image_predict / 255

    def make_prediction(self):

        self.set_preprocessing_image()

        prediction = []
        px = 30
        for letter in range(5):
            prediction.append(
                self.model.predict(
                    reshape(
                        self.image_predict[12:50, px:px + 20],
                        shape=(1, 38, 20, 1)
                    )
                )
            )
            px += 20

        p = [self.classes_captchSimbols[argmax(letter)] for letter in prediction]

        print(f'Predicted letters:\n{p}')
