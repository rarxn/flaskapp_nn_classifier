import python_speech_features
import scipy.io.wavfile
import numpy as np
import librosa


def trim_silence(audio, noise_threshold=150):
    """ Удаляет тишину в начале и конце полученных аудиоданных """
    start = None
    end = None

    for idx, point in enumerate(audio):
        if abs(point) > noise_threshold:
            start = idx
            break

    for idx, point in enumerate(audio[::-1]):
        if abs(point) > noise_threshold:
            end = len(audio) - idx
            break

    return audio[start:end]


def trim_silence_file(file_path):
    """ Перезаписывает оригинальный аудиофайл с отбросанной тишиной в начале и конце """
    audio, rate = librosa.load(file_path, sr=8000)
    b = np.abs(audio)
    trimmed_audio = trim_silence(audio, noise_threshold=b.max() / 75)
    scipy.io.wavfile.write(file_path, rate, trimmed_audio)


def padding(array, x_size, y_size):
    """ Дополняет матрицу данных до заданного размера """

    if array.shape[1] > y_size:
        print('to small second vall: ', array.shape)
        return array[:, 0:y_size]
    if array.shape[0] > x_size:
        print('to small first vall: ', array.shape)
        return array[0:x_size, :]
    height = array.shape[0]
    width = array.shape[1]
    a = (x_size - height) // 2
    aa = x_size - a - height

    b = (y_size - width) // 2
    bb = y_size - b - width
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def mfcc_psf(xf, sr, size):
    """ Возвращает mfcc признаки в едином формате """
    mfcc_feat = python_speech_features.mfcc(xf, sr, numcep=13)
    mfcc_feat = padding(mfcc_feat, size, 13)
    return mfcc_feat


def model_predict(model, filename):
    """ Функция формирования прогноза на базе модели нейронной сети, используя аудиофайл с входными данными """
    (xf, sr) = librosa.load(filename, sr=8000)
    mfcc_feat = python_speech_features.mfcc(xf, sr, numcep=13)
    mfcc_feat = padding(mfcc_feat, 100, 13)
    sample = np.expand_dims(mfcc_feat, 0)
    sample = np.expand_dims(sample, -1)
    predict = np.argmax(model.predict(sample, verbose=0))
    print(f'predicted: {predict}')
    return predict


def get_link(predict):
    """ switch case получение ссылки на основе прогноза """
    dictionary_predict_to_link = {
        0: 'https://vk.com', 1: 'https://yandex.ru/pogoda/',
        2: 'https://ssau.ru/', 3: 'https://github.com/',
        4: 'https://mail.google.com/', 5: 'https://www.youtube.com/',
        6: 'https://translate.google.com/', 7: 'https://ya.ru/',
        8: 'https://google.com/', 9: 'https://www.twitch.tv/'
    }
    return dictionary_predict_to_link[predict]


def get_prediction(audio, model):
    """ Возвращает результат обработки аудиоданных и прогнозирования моделью нейронной сети в виде ссылки """
    filename = './recordings/0.wav '
    audio.save(filename)
    trim_silence_file(filename)
    print('Файл с аудио получен и обработан успешно!')
    prediction = model_predict(model, filename)
    return get_link(prediction)
