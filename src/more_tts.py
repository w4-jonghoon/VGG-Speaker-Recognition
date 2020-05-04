import glob
import os
import re

import numpy as np
from tqdm import tqdm

MORE_TTS_RAW_PATH = '/nas0/poodle/speech_dataset/uncategorized/more-TTS-Jun2019/'
MORE_TTS_SPEAKERS = [
    'ko-KR-Wavenet-A_google', 'ko-KR-Wavenet-B_google',
    'ko-KR-Wavenet-C_google', 'ko-KR-Wavenet-D_google',
    'MAN_DIALOG_BRIGHT_kakao', 'MAN_READ_CALM_kakao',
    'WOMAN_DIALOG_BRIGHT_kakao', 'WOMAN_READ_CALM_kakao'
]
MORE_TTS_FILENAME_PATTERN = r'(ytn_)?(?P<utt_id>\d{5})_(?P<spk_id>[\w\-_]+).wav'
MORE_TTS_FILENAME_PATTERN_STR = '{utt_id:05d}_{spk_id}.wav'

MAX_UTT = 1000


def prepare_data_all(convert_spkid_to_int=False):
    files = glob.iglob(os.path.join(MORE_TTS_RAW_PATH, '*.wav'))
    filename_pattern = re.compile(MORE_TTS_FILENAME_PATTERN)

    if convert_spkid_to_int:
        spkid_map = {spk_id: i for i, spk_id in enumerate(MORE_TTS_SPEAKERS)}

    path_spk_list = []

    for filename in tqdm(files):
        searched = filename_pattern.search(filename)
        if searched is None:
            continue

        filepath = os.path.join(MORE_TTS_RAW_PATH, filename)
        if convert_spkid_to_int:
            path_spk_list.append(
                (filepath, spkid_map[searched.group('spk_id')]))
        else:
            path_spk_list.append((filepath, searched.group('spk_id')))
    return path_spk_list


def get_moretts_datalist():
    path_spk_list = prepare_data_all(convert_spkid_to_int=True)
    path_spk_list = np.array(path_spk_list)

    train_list, val_list = np.split(path_spk_list,
                                    [int(.8 * path_spk_list.shape[0])])
    return (train_list[:, 0], train_list[:, 1], val_list[:, 0], val_list[:, 1])


if __name__ == '__main__':
    trnlist, trnlb, vallist, vallb = get_moretts_datalist()
