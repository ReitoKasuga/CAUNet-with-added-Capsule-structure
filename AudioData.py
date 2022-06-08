import torch
from torch.utils.data import Dataset
from preprocess import SignalToFrames, ToTensor
import soundfile as sf
import numpy as np
import random
import h5py
import glob
import os
from scipy.io.wavfile import read

class TrainingDataset(Dataset):
    r"""Training dataset."""

    def __init__(self, train_file_list_path_noisy, train_file_list_path_clean, frame_size=512, frame_shift=256, nsamples=64000):
        self.file_list_noisy = os.listdir(train_file_list_path_noisy)
        self.file_list_clean = os.listdir(train_file_list_path_clean)
        self.file_path_noisy = train_file_list_path_noisy
        self.file_path_clean = train_file_list_path_clean

        self.nsamples = nsamples
        self.get_frames = SignalToFrames(frame_size=frame_size,
                                         frame_shift=frame_shift)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.file_list_noisy)

    def __getitem__(self, index):
        filename_noisy = self.file_path_noisy + '/' + self.file_list_noisy[index]
        filename_clean = self.file_path_clean + '/' + self.file_list_clean[index]

        _, reader_noisy = read(filename_noisy)
        _, reader_clean = read(filename_clean)

        feature = reader_noisy
        label = reader_clean

        size = feature.shape[0]
        start = random.randint(0, max(0, size + 1 - self.nsamples))
        feature = feature[start:start + self.nsamples]
        label = label[start:start + self.nsamples]

        feature = np.reshape(feature, [1, -1])  # [1, sig_len]
        label = np.reshape(label, [1, -1])  # [1, sig_len]

        feature = self.get_frames(feature)  # [1, num_frames, sig_len]

        feature = self.to_tensor(feature)  # [1, num_frames, sig_len]
        label = self.to_tensor(label)  # [1, sig_len]

        return feature, label

class EvalDataset(Dataset):
    r"""Evaluation dataset."""

    def __init__(self, evaluation_file_list_path_noisy, evaluation_file_list_path_clean, frame_size=512, frame_shift=256):
        self.file_list_noisy = os.listdir(evaluation_file_list_path_noisy)
        self.file_list_clean = os.listdir(evaluation_file_list_path_clean)
        self.file_path_noisy = evaluation_file_list_path_noisy
        self.file_path_clean = evaluation_file_list_path_clean
        self.get_frames = SignalToFrames(frame_size=frame_size,
                                         frame_shift=frame_shift)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.file_list_noisy)

    def __getitem__(self, index):
        filename_noisy = self.file_path_noisy + '/' + self.file_list_noisy[index]
        filename_clean = self.file_path_clean + '/' + self.file_list_clean[index]

        _, reader_noisy = read(filename_noisy)
        _, reader_clean = read(filename_clean)

        feature = reader_noisy
        label = reader_clean

        feature = np.reshape(feature, [1, 1, -1])  # [1, 1, sig_len]

        feature = self.get_frames(feature)  # [1, 1, num_frames, frame_size]

        feature = self.to_tensor(feature)  # [1, 1, num_frames, frame_size]
        label = self.to_tensor(label)  # [sig_len, ]

        return feature, label

class TrainCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            feat_dim = batch[0][0].shape[-1]  # frame_size
            label_dim = batch[0][1].shape[-1]  # sig_len

            feat_nchannels = batch[0][0].shape[0]  # 1
            label_nchannels = batch[0][1].shape[0]  # 1

            # sorted by sig_len for label
            sorted_batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)
            # (num_frames, sig_len)
            lengths = list(map(lambda x: (x[0].shape[1], x[1].shape[1]), sorted_batch))

            padded_feature_batch = torch.zeros((len(lengths), feat_nchannels, lengths[0][0], feat_dim))
            padded_label_batch = torch.zeros((len(lengths), label_nchannels, lengths[0][1]))
            lengths1 = torch.zeros((len(lengths),), dtype=torch.int32)

            for i in range(len(lengths)):
                padded_feature_batch[i, :, 0:lengths[i][0], :] = sorted_batch[i][0]
                padded_label_batch[i, :, 0:lengths[i][1]] = sorted_batch[i][1]
                lengths1[i] = lengths[i][1]

            return padded_feature_batch, padded_label_batch, lengths1
        else:
            raise TypeError('`batch` should be a list.')

class EvalCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            return batch[0][0], batch[0][1]
        else:
            raise TypeError('`batch` should be a list.')