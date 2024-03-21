from crypt import methods
from flask import Flask, request, jsonify, render_template, redirect, url_for
from waitress import serve
from model.model import predict_behavior
import os
from werkzeug.utils import secure_filename, safe_join

 
app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = '/uploads'
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS={'csv'}

current_dir = os.path.dirname(__file__)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST', 'GET'])
def predict():
    uploaded = False
    pred = False
    prediction = 0
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file!'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_dir, app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded=True
            # use safe_join to prevent directory traversal attacks
            uploads_dir = app.config['UPLOAD_FOLDER']
            allowed_paths = [safe_join(uploads_dir, f) for f in os.listdir(uploads_dir)]
            # files = [os.path.basename(path) for path in allowed_paths]  # Get filenames only
            files = os.listdir(uploads_dir)
            # submit_action = request.form.get('submit_action')
            # print(submit_action)
            # if submit_action == 'Test':
            #     pred = True
            #     prediction = predict_behavior(request.files['test_file'].filename)
            #     print(f"The prediction is {prediction}")
            #     return render_template('upload.html', filename=filename, uploaded=uploaded, files=files, pred=pred, prediction=prediction)
            return render_template('upload.html', filename=filename, uploaded=uploaded, files=files)

            # redirect(url_for('/', filename=filename))
    return render_template('upload.html')

@app.route('/display_result', methods=['POST'])
def display_result():
    # submit_action = request.form.get('submit_action')
    # print(submit_action)
    # if submit_action == 'Test':
    #     pred = True
    #     prediction = predict_behavior(request.files['test_file'].filename)
    #     print(f"The prediction is {prediction}")
    #     return render_template('display_result.html', pred=pred, prediction=prediction)
    return render_template('display_result')

@app.route('/dummy', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file is uploaded
        if 'file' not in request.files:
            return 'No file uploaded!'
        file = request.files['file']
        # Validate filename and extension
        if file.filename == '':
            return 'No selected file!'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_dir, app.config['UPLOAD_FOLDER'], filename)
            print(current_dir)
            file.save(filepath)
            # Process the uploaded file here (optional)
            # ...
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload.html')
        
@app.route('/uploaded_file/<filename>')
def uploaded_file(filename):
  return render_template('display_file.html', filename=filename)

 
if __name__ == '__main__':
    serve(app, host="127.0.0.1", port=8000)
