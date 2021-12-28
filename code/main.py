import argparse
import os
import shutil
import warnings

import ipdb
import torch
from logger import get_logger
from data_processing import load_user_data,load_news_data
from utils import get_clusters
from NPA import NPA
from evaluation import validate
from test import test
from Train import train
from Data_load import get_embedding,news_load,load_behavior_data
from predict import predict_result
from predict import predict_result
import numpy as np
warnings.filterwarnings('ignore')
global best_acc
best_acc=100

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch NPA implementation')
    parser.add_argument('--dataset', metavar='DATASET',type=str,
                        help='path to train dataset')
    parser.add_argument('--train_news_data', type=str, metavar='PATH',
                        help='path to train dataset')
    parser.add_argument('--dev_news_data', type=str, metavar='PATH',
                        help='path to valid dataset')
    parser.add_argument('--test_news_data', type=str, metavar='PATH',
                        help='path to test dataset')
    parser.add_argument('--train_behavior_data', type=str, metavar='PATH',
                        help='path to train dataset')
    parser.add_argument('--dev_behavior_data', type=str, metavar='PATH',
                        help='path to valid dataset')
    parser.add_argument('--test_behavior_data', type=str, metavar='PATH',
                        help='path to test dataset')
    parser.add_argument('--word_embedding_path',type=str,metavar='PATH',default='../../data/glove.42B.300d.txt',
                        help='word embedding path')
    parser.add_argument('-re', '--reuse', default='', type=str, metavar='PATH',
                        help='path to old model (default: none)')
    parser.add_argument('--print_freq', '-p', default=1000, type=int, metavar='N',
                        help='number of print_freq (default: 1000)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 100)')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--version',default='',type= str
                        ,help = 'distinguish the different versions of the model')
    parser.add_argument('--negtive_ratio',default=4,type = int,
                        help='the number of negtive samples')
    parser.add_argument('--max_news_length',default=30, type=int,
                        help='the max length of news title')
    parser.add_argument('--max_history',default=50,type=int,
                        help='the number of the max user historical news ')
    parser.add_argument('--word_embedding_size',default=300,type=int,
                        help='the dimension of word embedding')
    parser.add_argument('--cnn_filter_number',default=400,type=int,
                        help='the output channel of cnn')
    parser.add_argument('--cnn_window_size',default=3,type=int,
                        help='kernel size of cnn')
    parser.add_argument('--user_embedding_size',default=50,type=int,
                        help='dimension of user embedding')
    parser.add_argument('--word_preference_size',default=200,type=int,
                        help='MLP output dimention for user word preferences')
    parser.add_argument('--title_preference_size',default=200,type=int,
                        help='MLP output dimention for user word preferences')
    parser.add_argument('--test', default='', type=str, metavar='PATH',
                        help='path to old model to test (default: none)')
    parser.add_argument('--predict_file',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--train_user_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--train_behavior_data_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--train_behavior_label_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--dev_user_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--dev_behavior_data_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--dev_behavior_label_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--test_user_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--test_behavior_data_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--test_behavior_label_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--test_session_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--predict_user_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--predict_behavior_data_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--predict_behavior_label_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--predict_session_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--predict_impression_id_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--all_news_title_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--news2id_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    parser.add_argument('--user2id_cache',default='',type=str,metavar='PATH',
                        help='path to test file to be submitted')
    args = parser.parse_args()
    return args


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    filename = './checkpoint.pth.tar'.format(args.version)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './model_best.pth.tar'.format(args.version))


def main():
    global best_acc
    args = arg_parse()
    logger = get_logger(args.version)

    logger.info("==> logger build...")
    # Create dataloader
    logger.info('cuda is available: '+ str(torch.cuda.is_available()))
    logger.info("==> Creating dataloader...")
    train_behavior_data_dir = args.train_behavior_data
    dev_behavior_data_dir = args.dev_behavior_data
    test_behavior_data_dir = args.test_behavior_data
    train_news_data_dir = args.train_news_data
    dev_news_data_dir = args.dev_news_data
    test_news_data_dir = args.test_news_data


    user_data = load_user_data(train_cache = train_behavior_data_dir,
                                dev_cache = dev_behavior_data_dir,
                                test_cache = test_behavior_data_dir,)
    news_data = load_news_data(train_cache = train_news_data_dir,
                               dev_cache = dev_news_data_dir,
                               test_cache = test_news_data_dir)
    word_embedding, word2id = get_embedding(args.word_embedding_path)
    news2id, all_news_title = news_load(news_data,word2id,args)

    logger.info("==> dataload finish...")

    train_data_loader,dev_data_loader,test_data_loader,test_data,user2id,dev2_data_loader,dev2_data = load_behavior_data(user_data,news2id,args)

    logger.info("==> dataloader build...")


    model = NPA(all_news_title,
                user2id,
                word2id,
                word_embedding,
                args.negtive_ratio,
                args.max_news_length,
                args.max_history,
                args.word_embedding_size,
                args.cnn_filter_number,
                args.cnn_window_size,
                args.user_embedding_size,
                args.word_preference_size,
                args.title_preference_size)

    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    if args.predict_file:
        print('==> predict begin')
        if os.path.isfile(args.predict_file):
            logger.info("=> loading checkpoint '{}'".format(args.predict_file))
            checkpoint = torch.load(args.predict_file)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            logger.info('=> model best acc:{}'.format(best_acc))
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.predict_file, checkpoint['epoch']))
            with torch.no_grad():
                predict_result(model,test_data_loader,test_data)
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.predict_file))
        return

    if args.test:
        print('==> test begin')
        if os.path.isfile(args.test):
            logger.info("=> loading checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            logger.info('=> model best acc:{}'.format(best_acc))
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.test, checkpoint['epoch']))
            with torch.no_grad():
                test(model,dev2_data_loader,dev2_data)
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.test))
        return



    if args.reuse:
        if os.path.isfile(args.reuse):
            logger.info("=> loading checkpoint '{}'".format(args.reuse))
            checkpoint = torch.load(args.reuse)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            logger.info('=> model best acc:{}'.format(best_acc))
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.reuse, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.reuse))
            return

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    for epoch in range(args.start_epoch,args.epochs):

        train(train_data_loader,model,criterion,optimizer,epoch, args,logger)

        with torch.no_grad():
            logger.info('valid: ')
            acc = validate(dev_data_loader, model, criterion, epoch, args,logger)

        is_best = acc < best_acc
        best_acc = min(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': acc,
        }, is_best, args)

        with torch.no_grad():
            test(model,dev2_data_loader,dev2_data)
    with torch.no_grad():
        predict_result(model,test_data_loader,test_data)

if __name__ == '__main__':
    main()
