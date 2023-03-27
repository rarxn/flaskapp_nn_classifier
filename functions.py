import python_speech_features
import scipy.io.wavfile
import numpy as np
import librosa


def trim_silence(audio, noise_threshold=150):
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


def trim_silence_file(file_path, noise_threshold=150):
    audio, rate = librosa.load(file_path, sr=8000)
    b = np.abs(audio)
    trimmed_audio = trim_silence(audio, noise_threshold=b.max() / 75)
    scipy.io.wavfile.write(file_path, rate, trimmed_audio)


# def trim_silence_file(file_path, noise_threshold=150):
#     rate, audio = scipy.io.wavfile.read(file_path)
#     trimmed_audio = trim_silence(audio, noise_threshold=noise_threshold)
#     scipy.io.wavfile.write(file_path, rate, trimmed_audio)


def padding(array, xx, yy):
    if array.shape[1] > yy:
        print('to small second vall: ', array.shape)
        return array[:, 0:yy]
    if array.shape[0] > xx:
        print('to small first vall: ', array.shape)
        return array[0:xx, :]
    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def mfcc_psf(xf, sr, size):
    mfcc_feat = python_speech_features.mfcc(xf, sr, numcep=13)
    mfcc_feat = padding(mfcc_feat, size, 13)
    return mfcc_feat


def model_predict(model, filename):
    (xf, sr) = librosa.load(filename, sr=8000)
    mfcc_feat = python_speech_features.mfcc(xf, sr, numcep=13)
    mfcc_feat = padding(mfcc_feat, 100, 13)
    sample = np.expand_dims(mfcc_feat, 0)
    sample = np.expand_dims(sample, -1)
    predict = np.argmax(model.predict(sample, verbose=0))
    print(f'predicted: {predict}')
    return predict


def get_link(type):
    link = ''
    if type == 0:
        link = 'https://vk.com'
    elif type == 1:
        link = 'https://yandex.ru/pogoda/'
    elif type == 2:
        link = 'https://ssau.ru/'
    elif type == 3:
        link = 'https://github.com/'
    elif type == 4:
        link = 'https://mail.google.com/'
    elif type == 5:
        link = 'https://www.youtube.com/'
    elif type == 6:
        link = 'https://translate.google.com/'
    elif type == 7:
        link = 'https://ya.ru/'
    elif type == 8:
        link = 'https://google.com/'
    elif type == 9:
        link = 'https://www.twitch.tv/'
    return link
