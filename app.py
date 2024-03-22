from flask import Flask, render_template, request, flash
import mlflow
import os
from werkzeug.utils import secure_filename
import pandas as pd
from preprocess import *
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
current_dir = os.path.dirname(__file__)

def load_model():
    # logged_model = 'runs:/840db5c0df89494e80b28e9f65471f80/model'
    # loaded_model = mlflow.pyfunc.load_model(logged_model)
    loaded_model = joblib.load('model.pkl')
    return loaded_model

model = load_model()

def predict_behavior(data_path):
    cow = pd.read_csv(data_path)
    df = cow
    # make some transformations to the data
    df = convert_acc_units(df)
    df = simple_impute(df)
    df = encode_label_column(df)
    df = df.drop(columns=['TimeStamp_UNIX', 'TimeStamp_JST', 'Label', 'behavior'])
    # predict behavior
    model = load_model()
    behavior = set(model.predict(pd.DataFrame(df)))
    return behavior

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')

@app.route('/submit', methods=['POST', 'GET'])
def predict():
    show = False
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded'
        mfile = request.files['file']
        if mfile.filename == '':
            return 'No selected file'
        else:
            show = True
        filename = secure_filename(mfile.filename)
        filepath = os.path.join(current_dir, app.config['UPLOAD_FOLDER'], filename)
        mfile.save(filepath)

        b = predict_behavior(filepath)

        return render_template('index.html', prediction=b, filepath=filepath, show=show)
    return render_template('index.html', show=show)

if __name__ == '__main__':
    app.run(app, host="0.0.0.0", port=8000, debug=True)
