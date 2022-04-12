import yaml
import pathlib
import librosa as li
from ddsp.core import extract_loudness, extract_pitch
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
from scipy.io import wavfile
import soundfile as sf
import test
import math

PHONEMES = ['sil', 'b','d','f','g','h','j','k','l','m','n','p','r','s','t','v','w','z','zh','ch','sh','th','dh','ng','y','ae','ei','e','ii','i','ai','a','ou','u','ao','uu','oi','au','eo','er','oo']
PHONEME2ID={}
for i,p in enumerate(PHONEMES):
  PHONEME2ID[p] = i

def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(f, sampling_rate, block_size, signal_length, wav_type, oneshot, **kwargs):
    x, sr = sf.read(f, dtype=wav_type)
    print("Reading {}: sr {}, shape {}, type {}".format(f, sr, x.shape, x.dtype))
    if(x.dtype == 'int16'):
        x = x / 32768.0
    if(x.ndim > 1):
        x = np.mean(x,axis=1)
    if(sr != sampling_rate):
        x = li.resample(y=x, orig_sr=sr, target_sr=sampling_rate)

    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    midi_path = str(f).replace('wav', 'mid')
    phoneme_path = str(f).replace('wav', 'txt')
    
    midi = test.load_midi(midi_path)
    starts = [m.start for m in midi]
    ends = [m.end for m in midi]
    pitches = [int(m.pitch) for m in midi]
    syllables = test.load_text(phoneme_path)

    assert(len(midi) == len(syllables))
    
    label_length = pitch.shape[0]
    phoneme_labels = np.zeros(label_length)
    tpb = block_size * 1.0 / sampling_rate
    cb = math.ceil(.05 / tpb)
    for i in range(len(midi)):
        start_frame = math.floor(midi[i].start * float(sampling_rate) / float(block_size))
        end_frame = math.ceil(midi[i].end * float(sampling_rate) / float(block_size))
        
        phonemes = syllables[i].split("_")
        N = len(phonemes)
        duration = (end_frame - start_frame)
        durations = np.zeros(N, dtype='int32')
        for i, p in enumerate(phonemes):
            if(PHONEME2ID[p] < PHONEME2ID['ae']):
                durations[i] = cb
        total_const_time = np.sum(durations)
        n_const = total_const_time / cb
        n_vowel = N - n_const
        N_per_vowel = (duration - total_const_time) / n_vowel
        for i, p in enumerate(phonemes):
            if(PHONEME2ID[p] >= PHONEME2ID['ae']):
                durations[i] = N_per_vowel
        idx = start_frame
        for i, p in enumerate(phonemes):
            phoneme_labels[idx:(idx + durations[i])] = PHONEME2ID[p]
            idx += durations[i]

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)
    phoneme_labels = phoneme_labels.reshape(x.shape[0], -1)

    return x, pitch, loudness, phoneme_labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))
        self.phonemes = np.load(path.join(out_dir, "phonemes.npy"))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])
        pho = torch.from_numpy(self.phonemes[idx])
        return s, p, l, pho


def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    files = get_files(**config["data"])
    pb = tqdm(files)

    signals = []
    pitchs = []
    loudness = []
    phonemes = []

    for f in pb:
        pb.set_description(str(f))
        x, p, l, pho = preprocess(f, **config["preprocess"])
        signals.append(x)
        pitchs.append(p)
        loudness.append(l)
        phonemes.append(pho)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)
    phonemes = np.concatenate(phonemes, 0).astype(np.int32)

    out_dir = config["preprocess"]["out_dir"]
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)
    np.save(path.join(out_dir, "phonemes.npy"), phonemes)


if __name__ == "__main__":
    main()