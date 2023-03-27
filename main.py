from flask import Flask, render_template, url_for, request
from flask_ngrok import run_with_ngrok
from tensorflow import keras
from functions import *

model = keras.models.load_model('main_model')
app = Flask(__name__)

run_with_ngrok(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        filename = './recordings/0.wav '
        audio = request.files.get('audio')
        audio.save(filename)
        trim_silence_file(filename, 50)
        print('Файл с аудио получен и обработан успешно!')
        type = model_predict(model, filename)
        return get_link(type)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()
