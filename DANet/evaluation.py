import museval
import soundfile as sf
import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')


# 窓関数(ハニング窓の平方根)
def square_root_of_hann(M, sym=False):
    w = scipy.signal.windows.hann(M, sym)
    w = np.sqrt(w)
    return w

# istft(逆短時間フーリエ変換)によりスペクトログラムから音声波形に戻す関数
def return_to_sound(separated, before_data, win_func=square_root_of_hann):
    before = list()
    for i in range(len(before_data)):
        before_i_1 = librosa.istft(before_data[i, 0, :, :].reshape((129, 100)), hop_length=64, win_length=256,
                                   window=win_func)
        before_i_2 = librosa.istft(before_data[i, 1, :, :].reshape((129, 100)), hop_length=64, win_length=256,
                                   window=win_func)
        before.append([before_i_1, before_i_2])

    before = np.asarray(before)

    mixed = list()
    for i in range(len(before_data)):
        before_i_1 = before_data[i, 0, :, :]
        before_i_2 = before_data[i, 1, :, :]
        mixed_i = librosa.istft(before_i_1 + before_i_2, hop_length=64, win_length=256, window=win_func)
        mixed.append(mixed_i)

    mixed = np.asarray(mixed)

    after = list()
    for i in range(len(separated)):
        after_i_1 = librosa.istft(separated[i, :, :, 0].reshape((129, 100)), hop_length=64, win_length=256,
                                  window=win_func)
        after_i_2 = librosa.istft(separated[i, :, :, 1].reshape((129, 100)), hop_length=64, win_length=256,
                                  window=win_func)
        after.append([after_i_1, after_i_2])

    after = np.asarray(after)

    return before, mixed, after


# NSDR、SIR、SARを求める関数
def evaluation(before, mixed, after):
    NSDR_list = list()
    SIR_list = list()
    SAR_list = list()
    which = list()

    for i in range(len(before)):
        reference = before[i]
        estimated = after[i]
        mix = np.asarray([mixed[i], mixed[i]])

        if not np.any(reference[0]):
            print("before[{},0] is zeros.".format(i))
            continue
        if not np.any(reference[1]):
            print("before[{},1] is zeros.".format(i))
            continue
        if not np.any(estimated[0]):
            print("after[{},0] is zeros.".format(i))
            continue
        if not np.any(estimated[1]):
            print("after[{},1] is zeros.".format(i))
            continue

        reference = reference.reshape(2, -1, 1)
        estimated = estimated.reshape(2, -1, 1)
        mix = mix.reshape(2, -1, 1)

        SDR, ISR, SIR, SAR, perm = museval.metrics.bss_eval(reference_sources=reference, estimated_sources=estimated)
        NSDR, _, _, _, _ = museval.metrics.bss_eval(reference_sources=reference, estimated_sources=mix)

        NSDR = SDR - NSDR

        temp_NSDR = NSDR
        temp_SIR = SIR
        temp_SAR = SAR

        reference = before[i]
        estimated = after[i]
        mix = np.asarray([mixed[i], mixed[i]])

        one = estimated[0]
        two = estimated[1]
        estimated = np.asarray([two, one])

        reference = reference.reshape(2, -1, 1)
        estimated = estimated.reshape(2, -1, 1)
        mix = mix.reshape(2, -1, 1)

        SDR, ISR, SIR, SAR, perm = museval.metrics.bss_eval(reference_sources=reference, estimated_sources=estimated)
        NSDR, _, _, _, _ = museval.metrics.bss_eval(reference_sources=reference, estimated_sources=mix)

        NSDR = SDR - NSDR

        if np.sum(NSDR) + np.sum(SIR) + np.sum(SAR) > np.sum(temp_NSDR) + np.sum(temp_SIR) + np.sum(temp_SAR):
            which.append(False)
            NSDR_list.append(NSDR)
            SIR_list.append(SIR)
            SAR_list.append(SAR)
        else:
            which.append(True)
            NSDR_list.append(temp_NSDR)
            SIR_list.append(temp_SIR)
            SAR_list.append(temp_SAR)

    NSDR_list = np.asarray(NSDR_list)
    SIR_list = np.asarray(SIR_list)
    SAR_list = np.asarray(SAR_list)
    which = np.asarray(which)

    return NSDR_list, SIR_list, SAR_list

# GNSDR、GSIR、GSARを求める関数
def final_eval(NSDR_list, SIR_list, SAR_list):
    GNSDR = np.mean(NSDR_list)
    GSIR = np.mean(SIR_list)
    GSAR = np.mean(SAR_list)

    return GNSDR, GSIR, GSAR

# istft(逆短時間フーリエ変換)により復元した音声を保存する関数
def save_sound(num, before, mixed, after):
    for i in range(len(before[num])):
        sf.write("before{}.wav".format(i+1),before[num,i],8000)

    sf.write("input_data.wav",mixed[num],8000)

    for i in range(len(after[num])):
        sf.write("after{}.wav".format(i+1),after[num,i],8000)


# 音声波形のグラフを表示して保存する
def save_sound_fig(num, before, mixed, after):
    for i in range(len(before[num])):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(before[num, i])) / 8000, before[num, i])
        fig.savefig("before{}.png".format(i+1))

    fig = plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(mixed[num])) / 8000, mixed[num])
    fig.savefig("input_data.png")

    for i in range(len(after[num])):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(after[num, i])) / 8000, after[num, i])
        fig.savefig("after{}.png".format(i + 1))