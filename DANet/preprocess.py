import pandas
import numpy as np
import scipy
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

# 窓関数(ハニング窓の平方根)
def square_root_of_hann(M, sym=False):
  w = scipy.signal.windows.hann(M, sym)
  w = np.sqrt(w)
  return w

# 前処理を行う関数
# 音声をファイルからロードしてリサンプリング、短時間フーリエ変換する
# 音声の順番に対応する、音の種類を表すラベルを生成する
def preprocess(labelpath, audiopath, win_func=square_root_of_hann):
    labels = pandas.read_csv(labelpath)
    labels = labels.sort_values(by=["category", "filename"])
    sound_names = np.asarray(labels.loc[:, "category"].value_counts().sort_index().index)
    wav_filenames = labels.loc[:, "filename"]
    wav_filenames = np.asarray(wav_filenames)

    sounds = list()
    sampling_rates = list()
    for name in wav_filenames:
        y, sr = librosa.load(audiopath + name)
        sounds.append(y)
        sampling_rates.append(sr)

    sounds_resampled = list()
    for i in range(len(sounds)):
        resampled = librosa.resample(sounds[i], sampling_rates[i], 8000)
        sounds_resampled.append(resampled)

    Fouriers = list()
    for i in range(len(sounds_resampled)):
        Fourier = librosa.stft(sounds_resampled[i], n_fft=256, hop_length=64, win_length=256,
                               window=win_func)
        Fouriers.append(Fourier)

    fig = plt.figure(figsize=(10 * 7, 4 * 8))
    for i in range(50):
        plt.subplot(8, 7, i + 1)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(Fouriers[40 * i])), y_axis='linear', x_axis='time',
                                 sr=8000)
        plt.colorbar(format='%+2.0f dB')

    fig.savefig("Fourier.png")
    Fouriers = np.asarray(Fouriers)

    return Fouriers, sound_names
