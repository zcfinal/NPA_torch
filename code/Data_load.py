import os.path
import pickle
import torch.utils.data as Data
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import scale
import torch
from nltk.tokenize import word_tokenize

def news_load(news_data,word2id,args):
    if os.path.isfile(args.all_news_title_cache):
        print('==> load news data from cache')
        all_news_title = np.load(args.all_news_title_cache)
        with open(args.news2id_cache,'rb') as handle:
            news2id = pickle.load(handle)
    else:
        print('==> load news data from scratch')
        train_data_title = generate_discrete_news(news_data.train['title'],word2id,args.max_news_length)
        dev_data_title = generate_discrete_news(news_data.dev['title'],word2id,args.max_news_length)
        test_data_title = generate_discrete_news(news_data.test['title'],word2id,args.max_news_length)
        all_news_title = np.array([[0]*args.max_news_length]+train_data_title+dev_data_title+test_data_title)
        all_news_id = news_data.train['news_id'].tolist()+news_data.dev['news_id'].tolist()+news_data.test['news_id'].tolist()
        news2id = {news:id for news,id in zip(all_news_id,range(1,1+len(all_news_id)))}
        with open(args.news2id_cache,'wb') as handle:
            pickle.dump(news2id,handle,protocol=pickle.HIGHEST_PROTOCOL)
        np.save(args.all_news_title_cache,all_news_title)

    return news2id,all_news_title

'''
data size:[datalen] content:str
'''
def generate_discrete_news(data, word2id, max_len=30):
    data_discrete = []
    for line in data:
        line = line.strip().lower()
        line = word_tokenize(line)
        line_discrete = []
        for word in line:
            if len(line_discrete)>=max_len:
                break
            if word in word2id.keys():
                line_discrete.append(word2id[word])
        if len(line_discrete)<max_len :
            line_discrete.extend([0]*(max_len-len(line_discrete)))
        data_discrete.append(line_discrete)
    return data_discrete


def get_embedding(FILE_PATH):
    embedding_word = dict()
    f = open(FILE_PATH, 'r', encoding='UTF-8')
    count = 0
    for line in f:
        values = line.split()
        word = values[0]
        print(word,values)
        coefs = np.array(values[1:], dtype='float32')
        embedding_word[word] = coefs
        count += 1
    f.close()
    print('==> GloVe vectors loaded...')
    print('==> embedding size: '+str(count))
    words = embedding_word.keys()
    word2id = {word:i for (word,i) in zip(words,range(1,len(words)+1))}
    id2word = {i:word for (word,i) in zip(words,range(1,len(words)+1))}
    embedding = dict()
    for word in word2id.keys():
        embedding[word2id[word]] = embedding_word[word]
    word_embedding = np.zeros((len(words)+1,300))
    for id in word2id.values():
        word_embedding[id,:] = embedding[id]
    return word_embedding , word2id


class behaviors(Data.Dataset):

    def __init__(self,data_,news2id,property,args):
        self.property=property
        self.negtive_ratio = args.negtive_ratio
        self.max_history = args.max_history
        self.news2id = news2id
        data = []
        label = []
        user_data = []
        if property=='train':
            for row in data_.iterrows():
                row = row[1]
                history_news = random.sample(row['history'],min(self.max_history,len(row['history'])))
                history_news = [news2id[news] for news in history_news] + [0]*(self.max_history-len(history_news))
                for positive_news in row['positive_news']:
                    candidate_news = [positive_news] + random.sample(row['negtive_news'],min(self.negtive_ratio,len(row['negtive_news'])))
                    candidate_news = [news2id[news] for news in candidate_news] + [0]*(self.negtive_ratio+1 -len(candidate_news))
                    target = [1]+[0]*self.negtive_ratio
                    order = list(range(len(target)))
                    random.shuffle(order)
                    shuffle_candidate_news = []
                    shuffle_target = []
                    for id in order:
                        shuffle_candidate_news.append(candidate_news[id])
                        shuffle_target.append(target[id])

                    data.append(shuffle_candidate_news+history_news)
                    label.append(shuffle_target)
                    user_data.append(row['user_id'])
        elif property=='test':
            self.session = []
            for row in data_.iterrows():
                self.session.append([len(data)])
                row = row[1]
                history_news = random.sample(row['history'],min(self.max_history,len(row['history'])))
                history_news = [news2id[news] for news in history_news] + [0]*(self.max_history-len(history_news))
                for positive_news in row['positive_news']:
                    data.append([news2id[positive_news]]+history_news)
                    label.append(1)
                    user_data.append(row['user_id'])
                for negtive_news in row['negtive_news']:
                    data.append([news2id[negtive_news]]+history_news)
                    label.append(0)
                    user_data.append(row['user_id'])
                self.session[-1].append(len(data))
        elif property=='predict':
            self.impression_id = []
            self.session = []
            for row in data_.iterrows():
                self.session.append([len(data)])
                row = row[1]
                self.impression_id.append(row['impression_id'])
                history_news = random.sample(row['history'],min(self.max_history,len(row['history'])))
                history_news = [news2id[news] for news in history_news] + [0]*(self.max_history-len(history_news))
                for news in row['impressions']:
                    data.append([news2id[news]]+history_news)
                    user_data.append(row['user_id'])
                self.session[-1].append(len(data))


        self.user_data = user_data
        self.user_set = set(user_data)
        self.data = torch.tensor(data)
        self.label = torch.tensor(label)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        if self.property=='predict':
            return self.data[item],self.user_data[item]
        else:
            return self.data[item],self.user_data[item],self.label[item]

    def generate_user_data(self,user2id):
        self.user_data = [user2id[user] for user in self.user_data]
        self.user_data = torch.tensor(self.user_data).unsqueeze(1)

    def cache_data(self,property,args):
        if property == 'train':
            self.user_data = self.user_data.numpy()
            np.save(args.train_user_cache,self.user_data)
            self.data = self.data.numpy()
            np.save(args.train_behavior_data_cache,self.data)
            self.label = self.label.numpy()
            np.save(args.train_behavior_label_cache,self.label)
        elif property == 'dev':
            self.user_data = self.user_data.numpy()
            np.save(args.dev_user_cache,self.user_data)
            self.data = self.data.numpy()
            np.save(args.dev_behavior_data_cache,self.data)
            self.label = self.label.numpy()
            np.save(args.dev_behavior_label_cache,self.label)
        elif property=='test':
            self.user_data = self.user_data.numpy()
            np.save(args.test_user_cache,self.user_data)
            self.data = self.data.numpy()
            np.save(args.test_behavior_data_cache,self.data)
            self.label = self.label.numpy()
            np.save(args.test_behavior_label_cache,self.label)
            self.session = np.array(self.session)
            np.save(args.test_session_cache,self.session)
        elif property=='predict':
            self.user_data = self.user_data.numpy()
            np.save(args.predict_user_cache,self.user_data)
            self.data = self.data.numpy()
            np.save(args.predict_behavior_data_cache,self.data)
            self.label = self.label.numpy()
            np.save(args.predict_behavior_label_cache,self.label)
            self.session = np.array(self.session)
            np.save(args.predict_session_cache,self.session)
            self.impression_id = np.array(self.impression_id)
            np.save(args.predict_impression_id_cache,self.impression_id)

class behaviors_cache(Data.Dataset):
    def __init__(self,property,args):
        self.property=property
        if property == 'train':
            data = np.load(args.train_behavior_data_cache)
            label = np.load(args.train_behavior_label_cache)
            user_data = np.load(args.train_user_cache)
            self.user_data = torch.tensor(user_data)
            self.data = torch.tensor(data)
            self.label = torch.tensor(label)
        elif property == 'dev':
            data = np.load(args.dev_behavior_data_cache)
            label = np.load(args.dev_behavior_label_cache)
            user_data = np.load(args.dev_user_cache)
            self.user_data = torch.tensor(user_data)
            self.data = torch.tensor(data)
            self.label = torch.tensor(label)
        elif property=='test':
            data = np.load(args.test_behavior_data_cache)
            label = np.load(args.test_behavior_label_cache)
            user_data = np.load(args.test_user_cache)
            self.user_data = torch.tensor(user_data)
            self.data = torch.tensor(data)
            self.label = torch.tensor(label)
            self.session = np.load(args.test_session_cache)
        elif property=='predict':
            data = np.load(args.predict_behavior_data_cache)
            label = np.load(args.predict_behavior_label_cache)
            user_data = np.load(args.predict_user_cache)
            self.user_data = torch.tensor(user_data)
            self.data = torch.tensor(data)
            self.label = torch.tensor(label)
            self.session = np.load(args.predict_session_cache)
            self.impression_id = np.load(args.predict_impression_id_cache)
    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        if self.property=='predict':
            return self.data[item],self.user_data[item]
        else:
            return self.data[item],self.user_data[item],self.label[item]

def get_training_data(data,news2id,args):
    train_loader = behaviors(data.train,news2id,'train',args)
    dev_loader = behaviors(data.dev,news2id,'train', args)
    dev2_loader = behaviors(data.dev,news2id,'test', args)
    test_loader = behaviors(data.test,news2id,'predict',args)
    user_set = train_loader.user_set
    user_set.update(dev_loader.user_set)
    user_set.update(test_loader.user_set)
    user2id = {user:id for id,user in enumerate(user_set)}
    train_loader.generate_user_data(user2id)
    dev_loader.generate_user_data(user2id)
    dev2_loader.generate_user_data(user2id)
    test_loader.generate_user_data(user2id)
    test_data =test_loader
    dev2_data = dev2_loader

    train_loader.cache_data('train',args)
    dev_loader.cache_data('dev',args)
    dev2_loader.cache_data('test',args)
    test_loader.cache_data('predict',args)
    with open(args.user2id_cache,'wb') as handle:
        pickle.dump(user2id,handle,protocol=pickle.HIGHEST_PROTOCOL)

    train_loader = Data.DataLoader(dataset=train_loader,
                                   num_workers=args.workers,
                                   batch_size=args.batch_size,
                                   shuffle=True,drop_last=True)
    dev_loader = Data.DataLoader(dataset=dev_loader,
                                   num_workers=args.workers,
                                   batch_size=args.batch_size,
                                   shuffle=False)
    dev2_loader = Data.DataLoader(dataset=dev2_loader,
                                  num_workers=args.workers,
                                  batch_size=args.batch_size,
                                  shuffle=False)
    test_loader = Data.DataLoader(dataset=test_loader,
                                  num_workers=args.workers,
                                  batch_size=150,
                                  shuffle=False)

    return train_loader,dev_loader,test_loader,test_data,user2id,dev2_loader,dev2_data

def get_training_data_from_cache(args):
    train_loader=behaviors_cache('train',args)
    dev_loader=behaviors_cache('dev',args)
    dev2_loader=behaviors_cache('test',args)
    test_loader=behaviors_cache('predict',args)
    test_data =test_loader
    dev2_data = dev2_loader
    with open(args.user2id_cache,'rb') as handle:
        user2id = pickle.load(handle)

    train_loader = Data.DataLoader(dataset=train_loader,
                                   num_workers=args.workers,
                                   batch_size=args.batch_size,
                                   shuffle=True,drop_last=True)
    dev_loader = Data.DataLoader(dataset=dev_loader,
                                 num_workers=args.workers,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    dev2_loader = Data.DataLoader(dataset=dev2_loader,
                                  num_workers=args.workers,
                                  batch_size=args.batch_size,
                                  shuffle=False)
    test_loader = Data.DataLoader(dataset=test_loader,
                                  num_workers=args.workers,
                                  batch_size=150,
                                  shuffle=False)
    return train_loader,dev_loader,test_loader,test_data,user2id,dev2_loader,dev2_data


def load_behavior_data(data,news2id,args):
    if os.path.isfile(args.train_user_cache):
        print('==> load behavior data from cache')
        return get_training_data_from_cache(args)
    else:
        print('==> load behavior data from scratch')
        return get_training_data(data,news2id,args)
