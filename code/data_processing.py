import os
import pickle
import random
import pandas as pd
import numpy as np
class Data:
    """
    Class for holding a number of datasets and metadata.
    """
    pass

def split_history(input_str):
    if pd.isna(input_str):
        return []
    input_str = input_str.strip().split()
    return input_str

def pick_positive(input_str):
    if pd.isna(input_str):
        return []
    input_str = input_str.strip().split()
    news = [item.split('-') for item in input_str]
    positive_news = [name for name,target in news if target=='1']
    return positive_news

def pick_negtive(input_str):
    if pd.isna(input_str):
        return []
    input_str = input_str.strip().split()
    news = [item.split('-') for item in input_str]
    negtive_news = [name for name,target in news if target=='0']
    return negtive_news

def load_user_data(train_cache = '../../data/train_behaviors.pickle',
              dev_cache = '../../data/dev_behaviors.pickle',
              test_cache = '../../data/test_behaviors.pickle',
              ):
    """
    Loads data from CSV files, processes and caches it in pickles for faster future loading.
    """

    if os.path.isfile(train_cache):
        # Load from cached files if they already exist
        train = pd.read_pickle(train_cache)
        dev = pd.read_pickle(dev_cache)
        test = pd.read_pickle(test_cache)
    else:
        datasets = []
        for kind in ['large_train','large_dev','large_test']:
            # Load original CSV file
            if kind=='large_test':
                csv_file = '../../data/%s/behaviors.tsv' % kind
                df = pd.read_csv(csv_file,sep='\t',header=None,
                                 names=['impression_id','user_id','time','history','impressions'])
                df['history'] = df['history'].apply(split_history)
                df['impressions'] = df['impressions'].apply(split_history)
            else:
                csv_file = '../../data/%s/behaviors.tsv' % kind
                df = pd.read_csv(csv_file,sep='\t',header=None,
                             names=['impression_id','user_id','time','history','impressions'])
                df['history'] = df['history'].apply(split_history)
                df['positive_news'] = df['impressions'].apply(pick_positive)
                df['negtive_news'] = df['impressions'].apply(pick_negtive)
                df.drop('impressions', axis=1, inplace=True)
            df.drop('time',axis=1, inplace=True)

            datasets.append(df)

        train, dev, test = datasets

        # Cache results in files
        train.to_pickle(train_cache)
        dev.to_pickle(dev_cache)
        test.to_pickle(test_cache)


    data = Data()
    data.__dict__.update({
        'train': train,
        'dev': dev,
        'test':test
    })
    return data


def load_news_data(train_cache = '../../data/train_news.pickle',
              dev_cache = '../../data/dev_news.pickle',
              test_cache = '../../data/test_news.pickle',
              ):
    """
    Loads data from CSV files, processes and caches it in pickles for faster future loading.
    """

    if os.path.isfile(train_cache):
        # Load from cached files if they already exist
        train = pd.read_pickle(train_cache)
        dev = pd.read_pickle(dev_cache)
        test = pd.read_pickle(test_cache)
    else:
        datasets = []
        for kind in ['large_train','large_dev','large_test']:
            # Load original CSV file
            csv_file = '../../data/%s/news.tsv' % kind
            df = pd.read_csv(csv_file,sep='\t',header=None,
                names=['news_id','category','subcategory','title','abstract','url','title entities'
                       ,'abstract entities'])
            df.drop('url', axis=1, inplace=True)
            df.drop('title entities', axis=1, inplace=True)
            df.drop('abstract entities',axis=1,inplace=True)
            df.drop('category',axis=1,inplace=True)
            df.drop('subcategory',axis=1,inplace=True)
            df.drop('abstract',axis=1,inplace=True)
            datasets.append(df)

        train, dev, test = datasets

        # Cache results in files
        train.to_pickle(train_cache)
        dev.to_pickle(dev_cache)
        test.to_pickle(test_cache)


    data = Data()
    data.__dict__.update({
        'train': train,
        'dev': dev,
        'test':test
    })
    return data



