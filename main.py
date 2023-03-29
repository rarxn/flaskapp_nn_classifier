from flask import Flask, render_template, url_for, request
from flask_ngrok import run_with_ngrok
from tensorflow import keras
from functions import get_prediction

# подгружаем обученную модель
model = keras.models.load_model('main_model')
# создание объекта веб-приложения и подключение сервиса ngrok
app = Flask(__name__)
run_with_ngrok(app)


# обработка маршрутизации
@app.route('/', methods=['GET', 'POST'])
def index():
    """ POST - обработка полученных данных от клиента, формирование прогноза
    GET - отображение html формы главной страницы
    """
    if request.method == "POST":
        audio = request.files.get('audio')
        return get_prediction(audio, model)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()
