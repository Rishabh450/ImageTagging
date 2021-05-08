from imageai.Classification import ImageClassification
import os
from flask import Flask, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import json

UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home_view():
    return "<h1>Welcome to Geeks for Geeks</h1>"


@app.route('/getTag', methods=['POST'])
def getTag():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        tagcount = request.form['count']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            execution_path = os.getcwd()

            prediction = ImageClassification()
            prediction.setModelTypeAsResNet50()
            prediction.setModelPath(os.path.join(execution_path, "resnet50_imagenet_tf.2.0.h5"))
            prediction.loadModel()

            predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, file.filename),
                                                                  result_count=tagcount)
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

            tag = list()
            for eachPrediction, eachProbability in zip(predictions, probabilities):
                tag.append(eachPrediction)
            if len(tag) > 0:
                profile_json = {
                    "status": 1,
                    "tag": tag

                }
                return jsonify(profile_json)
        else:
            profile_json = {
                "status": 0,
                "tag": "Invalid Format"

            }
            return jsonify(profile_json)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', '5000'), debug=True)
