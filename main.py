trainModel = False
predict = True

if trainModel:

    from os import path, mkdir

    if not path.exists('models'):
        mkdir('models')
    if not path.exists('outputs'):
        mkdir('outputs')


    from preprocessImages import PreprocessImages
    preprocessImages_obj = PreprocessImages()
    x_train, x_test, y_train, y_test = preprocessImages_obj.get_train_and_test_subset()

    from network_architecture import NetworkArchitecture
    network_architecture_obj = NetworkArchitecture()

    from fit_evaluate_model import FitEvaluateModel
    fit_evaluate_model_obj = FitEvaluateModel(
        network_architecture_obj.model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=200,
        batch_size=480
    )

elif predict:

    from predict import Predict
    imageTOpredict_str = '2mg87.png'
    predict_obj = Predict(imageTOpredict_str)
    predict_obj.make_prediction()