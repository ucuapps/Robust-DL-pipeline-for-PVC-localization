import os
import pandas as pd
import random
import numpy as np
from scipy.io import loadmat
import wfdb
import pytorch_lightning
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, lfilter
from scipy.signal import resample
import scipy.signal

from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal
import biosppy

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

def normalize(X):
    return (X - X.min()) / (X.max() - X.min())

def filter_ecg(ecg, sampling_rate=250, filter_type="FIR", filter_band="bandpass",
               filter_frequency=[3, 35], filter_order=12):
    if filter_type in ["FIR", "butter", "cheby1", "cheby2", "ellip", "bessel"]:
        order = int(filter_order * sampling_rate)
        filtered, _, _ = biosppy.tools.filter_signal(signal=ecg,
                                                     ftype=filter_type,
                                                     band=filter_band,
                                                     order=order,
                                                     frequency=filter_frequency,
                                                     sampling_rate=sampling_rate)
    else:
        raise ValueError('You can use only "FIR", "butter", "cheby1", "cheby2", "ellip", "bessel" filter types')

    return filtered


class ECGPeaksDatasetMIT(Dataset):
    SIGNAL_PATH_COLUMN = 'PathToData'
    CLASS_COLUMN = 'Label'

    def __init__(self, root_dir, df, transform=None, augmentation=None, config=None):
        """
        :param root_dir:
        :param df:
        :param transform:
        :param augmentation:
        :param config:
        """
        random.seed(42)
        if config is None:
            config = {}
        self.root_dir = root_dir
        self.df = df
        self.df[self.SIGNAL_PATH_COLUMN] = self.df[self.SIGNAL_PATH_COLUMN].apply(str)

        self.__create_labels()
        self.transform = transform
        self.augmentation = augmentation
        self.config = config
        self.signal_cache = {}

    def __create_labels(self):
        self.labels = self.df[self.CLASS_COLUMN].apply(self.__label_mapper)

    @staticmethod
    def __label_mapper(label):
        """
        Encode string label to the multi label vector
        Corresponding indexes of the disease:
        0: Normal
        1: SPB - Supraventricular Premature Block
        2: PVC - Premature ventricular complex
        :param label: string with names of the diseases
        :return: encoded multi label y
        """
        mapper = {'N': 0, 'PVC': 1  # , 'SPB': 2
                  }

        y = mapper[label]

        return y

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]

        path_to_file = str(info[self.SIGNAL_PATH_COLUMN])
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        y = y.squeeze()

        if path_to_file not in self.signal_cache.keys():
            path = os.path.join(self.root_dir, self.df['Dataset'][idx], path_to_file)
            raw_sign = wfdb.rdrecord(path)
            cur_x = np.asarray(raw_sign.p_signal[:, 0], dtype=np.float64)

            freq = raw_sign.fs
            res_ratio = 250 / raw_sign.fs
            num = int(cur_x.size * res_ratio)

            self.signal_cache[path_to_file] = signal.resample(cur_x, num=num)

        peak_idx = self.df.Location[idx]
        # peak_idx = peak_idx + random.randint(-15, 15)

        x = self.signal_cache[path_to_file][peak_idx - 120:peak_idx + 120]

        # x = butter_bandpass_filter(0.5, 40, 250, order=2)
        x = normalize(x)
        # x = torch.tensor(np.expand_dims(butter_bandpass_filter(x, 0.5, 30, 250, order=1), 0).copy(),
        #              dtype=torch.float64)
        x = np.expand_dims(x, 0)
        x = torch.tensor(x.copy(), dtype=torch.float64)
        if self.transform:
            x = self.transform(x)

        # x = x.view(-1)

        return x.float(), y.long()


class PQRSTDataset(Dataset):
    SIGNAL_PATH_COLUMN = 'PathToData'
    CLASS_COLUMN = 'Label'

    def __init__(self, root_dir, df, transform=None, augmentation=None, config=None):
        """
        :param root_dir:
        :param df:
        :param transform:
        :param augmentation:
        :param config:
        """
        random.seed(42)
        if config is None:
            config = {}
        self.root_dir = root_dir
        self.df = df
        self.__create_labels()
        self.transform = transform
        self.augmentation = augmentation
        self.config = config
        self.signal_cache = {}

    def __create_labels(self):
        self.labels = self.df[self.CLASS_COLUMN].apply(self.__label_mapper)

    @staticmethod
    def __label_mapper(label):
        """
        Encode string label to the multi label vector
        Corresponding indexes of the disease:
        0: Normal
        1: SPB - Supraventricular Premature Block
        2: PVC - Premature ventricular complex
        :param label: string with names of the diseases
        :return: encoded multi label y
        """

        mapper = {'Normal': 0, 'PVC': 1}#'SPB': 1, 'PVC': 2
        y = mapper[label]

        return y

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]

        path_to_file = info[self.SIGNAL_PATH_COLUMN]

        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        y = y.squeeze()

        r_location = self.df.Location[idx]
        p_location = self.df.p[idx]
        q_location = self.df.q[idx]
        s_location = self.df.s[idx]
        t_location = self.df.t[idx]

        if p_location == 0 or q_location - p_location > 100:
            p_location = q_location - 100
        else:
            p_location = p_location - 50

        if path_to_file not in self.signal_cache.keys():
            self.signal_cache[path_to_file] = np.asarray(
                loadmat(os.path.join(self.root_dir, 'TrainingSet', path_to_file))['ecg'], dtype=np.float64)

        x = self.signal_cache[path_to_file][p_location - 150:t_location + 150]
        # x = self.signal_cache[path_to_file][r_location-200:r_location + 200]

        x = signal.resample(x, 400)
        x = np.expand_dims(np.squeeze(x, 1), 0)
        x = torch.tensor(x, dtype=torch.float64)

        if self.transform:
            x = self.transform(x)

        # x = x.view(-1)

        return x.float(), y.long()


class ECGPeaksDataset(Dataset):
    SIGNAL_PATH_COLUMN = 'PathToData'
    CLASS_COLUMN = 'Label'

    def __init__(self, root_dir, df, transform=None, augmentation=None, config=None):
        """
        :param root_dir:
        :param df:
        :param transform:
        :param augmentation:
        :param config:
        """
        random.seed(42)
        if config is None:
            config = {}
        self.root_dir = root_dir
        self.df = df
        self.__create_labels()
        self.transform = transform
        self.augmentation = augmentation
        self.config = config
        self.signal_cache = {}

    def __create_labels(self):
        self.labels = self.df[self.CLASS_COLUMN].apply(self.__label_mapper)

    @staticmethod
    def __label_mapper(label):
        """
        Encode string label to the multi label vector
        Corresponding indexes of the disease:
        0: Normal
        1: SPB - Supraventricular Premature Block
        2: PVC - Premature ventricular complex
        :param label: string with names of the diseases
        :return: encoded multi label y
        """
        mapper = {'Normal': 0, 'PVC': 1#, 'SPB': 2
                  }
        
        y = mapper[label]
        
        return y

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]

        path_to_file = info[self.SIGNAL_PATH_COLUMN]

        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        y = y.squeeze()

        peak_idx = self.df.Location[idx]
        if path_to_file not in self.signal_cache.keys():
            self.signal_cache[path_to_file] = np.asarray(loadmat(os.path.join(self.root_dir, 'TrainingSet', path_to_file))['ecg'], dtype=np.float64)
            # print('start')
            # self.signal_cache[path_to_file] = np.expand_dims(filter_ecg(np.squeeze(np.asarray(
            #     loadmat(os.path.join(self.root_dir, 'TrainingSet', path_to_file))['ecg'], dtype=np.float64), 1)), 1)
            # print('finish')

        if self.augmentation:
            random_peak = peak_idx + random.randint(-10, 10)

            crop_prob = random.uniform(0, 1)
            down_prob = random.uniform(0, 1)
            noise_prob = random.uniform(0, 1)

            x = self.signal_cache[path_to_file][random_peak - 150:random_peak + 150]

            # if crop_prob > 0.5:
            # x = self.random_cropping(x, 15, np.random.randint(0,5)[0])
            # x = self.signal_downsampling(x, np.random.randint(0,3)[0])
            # x = self.add_noise(x, 0, np.random.uniform(low=0, high=0.05, size=1)[0])

            # if crop_prob > 0.5:
            #     x = self.random_cropping(x, 15, np.random.randint(0,5)[0])

            # if down_prob > 0.5:
            #     x = self.signal_downsampling(x, np.random.randint(0,3)[0])
            #
            # if noise_prob > 0.7:
            #     x = self.add_noise(x, 0, np.random.uniform(low=0, high=0.05, size=1)[0])

        else:
            x = self.signal_cache[path_to_file][peak_idx - 200:peak_idx + 200]
            x = signal.resample(x, 300)
            # x = x[len(x) - 200: len(x + 200)]
        x = np.expand_dims(np.squeeze(x, 1), 0)
        x = torch.tensor(x.copy(), dtype=torch.float64)

        if self.transform:
            x = self.transform(x)

        # x = x.view(-1)

        return x.float(), y.long()

    def signal_downsampling(self, inp_sig, take_every_th_element):
        downsampled_sig = inp_sig[::take_every_th_element]
        return signal.resample(downsampled_sig, len(inp_sig))

    def add_noise(self, inp_sig, noise_min_value, noise_max_value):
        noise = np.random.uniform(low=noise_min_value, high=noise_max_value, size=len(inp_sig))
        return inp_sig + noise

    def random_cropping(self, inp_sig, max_size_crop, num_crops):
        random_start_indexes = np.random.randint(low=20, high=len(inp_sig) - 20, size=num_crops)
        random_crop_sizes = np.random.randint(low=0, high=max_size_crop, size=num_crops)
        random_end_indexes = [random_start_indexes[i] + random_crop_sizes[i] for i in range(num_crops)]
        random_crops = list(zip(random_start_indexes, random_end_indexes))
        crop_mask = np.ones(len(inp_sig))
        for crop_start_ind, crop_finish_ind in random_crops:
            crop_mask[crop_start_ind:crop_finish_ind] = 0
        crop_mask = crop_mask.astype(bool)
        croped_sig = inp_sig[crop_mask]
        croped_sig = signal.resample(croped_sig, len(inp_sig))
        return croped_sig


class GaussianPeakDatasetMultilabel(Dataset):
    SIGNAL_PATH_COLUMN = 'PathToData'
    CLASS_COLUMN = 'Label'

    def __init__(self, root_dir, df, transform=None, augmentation=None, config=None):
        """
        :param root_dir:
        :param df:
        :param transform:
        :param augmentation:
        :param config:
        """
        random.seed(42)
        if config is None:
            config = {}
        self.root_dir = root_dir
        self.df = df
        self.__create_labels()
        self.transform = transform
        self.augmentation = augmentation
        self.config = config
        self.signal_cache = {}

    def __create_labels(self):
        self.labels = self.df[self.CLASS_COLUMN].apply(self.__label_mapper)

    @staticmethod
    def __label_mapper(label):
        """
        Encode string label to the multi label vector
        Corresponding indexes of the disease:
        0: Normal
        1: SPB - Supraventricular Premature Block
        2: PVC - Premature ventricular complex
        :param label: string with names of the diseases
        :return: encoded multi label y
        """
        mapper = {'Normal': 0, 'SPB': 1, 'PVC': 2}

        y = mapper[label]

        return y

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]

        path_to_file = info[self.SIGNAL_PATH_COLUMN]

        peak_idx = self.df.Location[idx]
        generated_int = random.randint(-50, 50)
        random_peak = peak_idx + generated_int

        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        label = int(label.squeeze())

        if path_to_file not in self.signal_cache.keys():
            self.signal_cache[path_to_file] = np.asarray(
                loadmat(path_to_file)['ecg'], dtype=np.float64)

        if self.augmentation:
            x = self.signal_cache[path_to_file][random_peak - 200:random_peak + 200]
            y = torch.zeros(self.labels.max()+1, len(x))
            if label:
                gaussian_ind = int((len(x) / 2) + generated_int)
                gaussian_channel = self.generate_gaussian(gaussian_ind, len(x))
                print(label, y.shape)
                y[label] = torch.from_numpy(gaussian_channel) / gaussian_channel.max()
        else:
            x = self.signal_cache[path_to_file][peak_idx - 200:peak_idx + 200]
            y = torch.zeros(self.labels.max()+1, len(x))

            if label:
                gaussian_ind = int((len(x) / 2))
                gaussian_channel = self.generate_gaussian(gaussian_ind, len(x))
                y[label] = torch.from_numpy(gaussian_channel) / gaussian_channel.max()

        x = np.expand_dims(np.squeeze(x, 1), 0)
        x = torch.tensor(x, dtype=torch.float64)

        if self.transform:
            x = self.transform(x)

        return x.float(), y

    def generate_gaussian(self, gaussian_ind, signal_len, sigma=3):
        gaussian_channel = np.zeros(signal_len, dtype=float)
        gaussian_channel[gaussian_ind] = 1
        gaussian_sig = gaussian_filter1d(gaussian_channel, sigma)
        return gaussian_sig


class GaussianPeakDatasetBinary(Dataset):
    SIGNAL_PATH_COLUMN = 'PathToData'
    CLASS_COLUMN = 'Label'

    def __init__(self, root_dir, df, transform=None, augmentation=None, config=None):
        """
        :param root_dir:
        :param df:
        :param transform:
        :param augmentation:
        :param config:
        """
        random.seed(42)
        if config is None:
            config = {}
        self.root_dir = root_dir
        self.df = df
        self.__create_labels()
        self.transform = transform
        self.augmentation = augmentation
        self.config = config
        self.signal_cache = {}

    def __create_labels(self):
        self.labels = self.df[self.CLASS_COLUMN].apply(self.__label_mapper)

    @staticmethod
    def __label_mapper(label):
        """
        Encode string label to the multi label vector
        Corresponding indexes of the disease:
        0: Normal
        1: SPB - Supraventricular Premature Block
        2: PVC - Premature ventricular complex
        :param label: string with names of the diseases
        :return: encoded multi label y
        """
        mapper = {'Normal': 0, 'SPB': 1, 'PVC': 2}

        y = mapper[label]

        return y

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]

        path_to_file = info[self.SIGNAL_PATH_COLUMN]

        peak_idx = self.df.Location[idx]
        generated_int = random.randint(-370, 370)
        random_peak = peak_idx + generated_int

        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        label = int(label.squeeze())

        if path_to_file not in self.signal_cache.keys():
            self.signal_cache[path_to_file] = np.asarray(
                loadmat(os.path.join(self.root_dir, 'TrainingSet', path_to_file))['ecg'], dtype=np.float64)

        if self.augmentation:
            x = self.signal_cache[path_to_file][random_peak - 500:random_peak + 500]
            # print()
            gaussian_ind = int((len(x) / 2) - generated_int)
            # print("Peak ind ", peak_idx)
            # print("Gaussian ind ", gaussian_ind)
            # print("Generated int", generated_int)

            gaussian_channel = self.generate_gaussian(gaussian_ind, len(x))
            if label:
                y = torch.from_numpy(gaussian_channel) / gaussian_channel.max()
            else:
                y = torch.zeros(len(x), dtype=torch.float)
        else:
            x = self.signal_cache[path_to_file][peak_idx - 500:peak_idx + 500]
            gaussian_ind = int((len(x) / 2))
            gaussian_channel = self.generate_gaussian(gaussian_ind, len(x))
            if label:
                y = torch.from_numpy(gaussian_channel) / gaussian_channel.max()
            else:
                y = torch.zeros(len(x), dtype=torch.float)
        x = np.expand_dims(np.squeeze(x, 1), 0)
        x = torch.tensor(x, dtype=torch.float64)

        if self.transform:
            x = self.transform(x)

        return x.float(), y.float()

    def generate_gaussian(self, gaussian_ind, signal_len, sigma=33):
        gaussian_channel = np.zeros(signal_len, dtype=float)
        gaussian_channel[gaussian_ind] = 1
        gaussian_sig = gaussian_filter1d(gaussian_channel, sigma)
        return gaussian_sig


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    def __train_val_split(df):
        train_data, test_data = train_test_split(df, test_size=0.2,
                                                 random_state=42, stratify=df['Label'])
        return train_data, test_data

    root_dir = '/datasets/ecg/china_challenge'
    df = pd.read_csv(os.path.join(root_dir, 'icbeb_peaks.csv'))
    df = df[df['Label']!= 'SPB'].reset_index()
    val_df = df[df['PathToData'].isin(['A09.mat', 'A02.mat'])]
    # val_df[val_df['Label'] == 'Normal'] = val_df[val_df['Label'] == 'Normal']#[:5000]

    train_df = df[~df['PathToData'].isin(['A09.mat', 'A02.mat'])]
    # train_df[train_df['Label'] == 'Normal'] = train_df[train_df['Label'] == 'Normal']#[:50000]

    train_df.reset_index(inplace=True)
    val_df.reset_index(inplace=True)

    ds = ECGPeaksDataset(root_dir, val_df, transform=None, augmentation=True, config=None)

    print(len(ds))
    for i in range(3):
        x, y = ds[i]
        print(y.shape)
        print(x.shape)
        print()