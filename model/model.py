import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import subprocess
import joblib
import os

from preprocess import *
# export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"

# MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# subprocess.Popen(["mlflow", "ui", "--port", "8080", "--backend-store-uri", MLFLOW_TRACKING_URI])

mlflow.set_tracking_uri('https://cattle-behavior-classifier.onrender.com/')

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'JapaneseBlackBeefData/')

cow1 = pd.read_csv(os.path.join(data_path, 'cow1.csv'))
cow2 = pd.read_csv(os.path.join(data_path, 'cow2.csv'))
cow3 = pd.read_csv(os.path.join(data_path, 'cow3.csv'))
cow4 = pd.read_csv(os.path.join(data_path, 'cow4.csv'))
cow5 = pd.read_csv(os.path.join(data_path, 'cow5.csv'))
cow6 = pd.read_csv(os.path.join(data_path, 'cow6.csv'))


# # mlflow will create an experiment if it doesn't exist
experiment_name = "behavior-classification-experiment"
# run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

# print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
# conf.get_default().auth_token = getpass.getpass()
# port=5000
# public_url = ngrok.connect(port).public_url
# print(f' * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"')


params = {"max_depth": 2, "random_state": 42}
model = RandomForestClassifier(**params)

def experiment(df):
    df = convert_acc_units(df)
    df = simple_impute(df)
    df = encode_label_column(df)

    mlflow.set_experiment(experiment_name=experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # mlflow.tensorflow.autolog(disable=True)
    # mlflow.keras.autolog(disable=True)
    # mlflow.autolog()

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        features = ['AccX', 'AccY', 'AccZ', 'AccMag']
        target_variable = 'behavior'

        X = df[features]
        y = df[target_variable]

        # standardize the dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        trainingd = mlflow.data.from_pandas(X_train, source="data.csv")
        testd = mlflow.data.from_pandas(X_test, source="data.csv")
        mlflow.log_input(trainingd, context="trainig")
        mlflow.log_input(testd, context="testing")

        # define and train the random forest model
        # model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        mlflow.log_params(params)

        # make predictions on training data
        y_pred = model.predict(X_test)

        # evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)


        # print evaluation metrics
        print(f'Accuracy: {accuracy}')
        print(f'Mean Squared Error: {mse}')
        print(f'Mean Absolute Error: {mae}')
        print(f'R-squared: {r2}')

        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     artifact_path="sklearn-model",
        #     input_example=X_scaled,
        #     registered_model_name="sk-learn-random-forest-class-model",
        # )
        #  Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        # model signature
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=signature, 
                                 serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
                                 registered_model_name="sk-learn-random-forest-model")

    joblib.dump(model, 'model.pkl')
    # end run
    mlflow.end_run()


def train():
    experiment(cow1)
    # experiment(cow2)
    # experiment(cow3)
    # experiment(cow4)

# train()

def test():
    # check if model exists, if not create it
    if not os.path.isfile(os.path.join(current_dir, '..', 'model.pkl')):
        train()

    # Load the saved model
    model = joblib.load(os.path.join(current_dir, '..', 'model.pkl'))
    # use the loaded model to make predictions
    logged_model = 'runs:/840db5c0df89494e80b28e9f65471f80/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    df = cow1
    # make some transformations to the data
    df = convert_acc_units(df)
    df = simple_impute(df)
    df = encode_label_column(df)
    df = df.drop(columns=['TimeStamp_UNIX', 'TimeStamp_JST', 'Label', 'behavior'])
    # predict behavior
    behavior = set(loaded_model.predict(pd.DataFrame(df)))
    return behavior


def predict_behavior(df):
    # check if model exists, if not create it
    if not os.path.isfile(os.path.join(current_dir, '..', 'model.pkl')):
        train()

    # Load the saved model
    model = joblib.load(os.path.join(current_dir, '..', 'model.pkl'))
    # use the loaded model to make predictions

    # make some transformations to the data
    df = convert_acc_units(df)
    df = simple_impute(df)
    df = encode_label_column(df)
    # predict behavior
    behavior = set(model.predict(pd.DataFrame(df)))
    return behavior