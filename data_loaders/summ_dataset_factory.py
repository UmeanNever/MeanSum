# summ_dataset_factory.py

"""
Class to create SummDataset instances.

In part here in a separate file to avoid circular imports
"""

from data_loaders.amazon_dataset import AmazonDataset
from data_loaders.yelp_dataset import YelpDataset
from data_loaders.my_dataset import MyDataset


class SummDatasetFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get(name):
        if name == 'amazon':
            return AmazonDataset()
        elif name == 'yelp':
            return YelpDataset()
        elif name in ['my', 'schwab', 'twitter']:
            return MyDataset(name)
        else:
            print("Dataset name not defined!!")
            return MyDataset(name)
