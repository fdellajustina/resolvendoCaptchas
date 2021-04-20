from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt

class FitEvaluateModel:

    def __init__(self, model, x_train, y_train, x_test, y_test, epochs, batch_size):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size

        self.fitModel()

        self.saveModel()
        self.savePlot_modelHistory()

        self.evaluateModel()

    def fitModel(self):

        history = self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )

        hist_csv_file = 'outputs/history.csv'
        with open(hist_csv_file, mode='w') as f:
            DataFrame(history.history).to_csv(f)

    def evaluateModel(self):

        loss, acc = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=2
        )
        print(f'Acurácia: {acc}')

    def saveModel(self):

        # The '.h5' extension indicates that the model should be saved to HDF5.
        self.model.save('models/captcha.h5')


    def savePlot_modelHistory(self):

        history = read_csv('outputs/history.csv')

        plt.plot(history['accuracy'], label='accuracy')
        plt.plot(history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.savefig('outputs/history.png')
