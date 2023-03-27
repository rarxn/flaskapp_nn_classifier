# unsupported and must be removed
# unsupported and must be removed
# unsupported and must be removed
# unsupported and must be removed
# unsupported and must be removed

import pyaudio
import wave

FILES = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
RECORD_SECONDS = 2
WAVE_OUTPUT_FILEPATH = "./recordings/"


def get_numFiles():
    return str(FILES)


def add_numFiles():
    global FILES
    FILES += 1


def record():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    wave_output_filename = WAVE_OUTPUT_FILEPATH + get_numFiles() + '.wav'
    # add_numFiles()
    wf = wave.open(wave_output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return wave_output_filename
