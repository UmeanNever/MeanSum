from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from data_loaders.summ_dataset import SummReviewDataset, SummDataset
from project_settings import HParams, DatasetConfig
from data_loaders.extra_word_filter import ExtraWordFilter

import pandas as pd
import numpy as np
from utils import load_file, save_file
from collections import defaultdict


class MyPytorchDataset(Dataset):
    def __init__(self, name='my', split='', n_reviews=None, subset=None, **kwargs):
        self.ds_conf = DatasetConfig(name)
        self.reviews = []
        self.key_to_id = {}
        self.id_to_key = defaultdict(str)
        if split == 'lm':
            data = np.load(self.ds_conf.npy_path)
            for item in data:
                self.reviews.append(item[:n_reviews])
            if subset:
                new_size = max(int(len(self.reviews) * subset), 1)
                self.reviews = self.reviews[:new_size]
        else:
            df = pd.read_csv(self.ds_conf.csv_path)
            for i, column_name in enumerate(df.columns[1:]):
                value = df[column_name].values
                self.reviews.append(value[:n_reviews])
                self.key_to_id[column_name] = i
                self.id_to_key[i] = column_name
            if subset:
                new_size = max(int(len(self.reviews) * subset), 1)
                self.reviews = self.reviews[:new_size]
        self.n = len(self.reviews)
        if split != 'lm':
            self.extra_word_filter = ExtraWordFilter()
            self.filtered_reviews = self.extra_word_filter.fit_transform(self.reviews, no_above=0.1, no_below=1)
        else:
            self.filtered_reviews = self.reviews

    def __getitem__(self, idx):
        texts = SummDataset.concat_docs(self.reviews[idx], edok_token=True)
        filtered_texts = SummDataset.concat_docs(self.filtered_reviews[idx], edok_token=True)
        return texts, 1, {'Topic': self.id_to_key[idx], 'Filtered_Text': filtered_texts}

    def __len__(self):
        return self.n


class MyDataset(SummReviewDataset):
    def __init__(self, name):
        super(MyDataset, self).__init__()
        self.name = name
        self.conf = DatasetConfig(name)
        self.n_ratings_labels = 1
        self.reviews = None
        self.subwordenc = load_file(self.conf.subwordenc_path)

    ####################################
    #
    # Utils
    #
    ####################################
    def get_data_loader(self, split='train',
                        n_docs=20, n_docs_min=None, n_docs_max=None,
                        subset=None, seed=0, sample_reviews=False,
                        category=None,  # for compatability with AmazonDataset, which filters in AmazonPytorchDataset
                        batch_size=64, shuffle=True, num_workers=4):
        """
        Return iterator over specific split in dataset
        """
        ds = MyPytorchDataset(name=self.name, split=split,
                              n_reviews=n_docs, n_reviews_min=n_docs_min, n_reviews_max=n_docs_max,
                              subset=subset, seed=seed, sample_reviews=sample_reviews,
                              item_max_reviews=self.conf.item_max_reviews)

        if n_docs_min and n_docs_max:
            loader = DataLoader(ds, batch_sampler=VariableNDocsSampler(ds), num_workers=num_workers)
        else:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader


