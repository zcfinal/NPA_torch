#!/bin/bash

train_news_data=../../data/large_train/news.pickle
dev_news_data=../../data/large_dev/news.pickle
test_news_data=../../data/large_test/news.pickle
train_behavior_data=../../data/large_train/behaviors.pickle
dev_behavior_data=../../data/large_dev/behaviors.pickle
test_behavior_data=../../data/large_test/behaviors.pickle
train_user_cache=../../data/large_train/user.npy
train_behavior_data_cache=../../data/large_train/behavior_data.npy
train_behavior_label_cache=../../data/large_train/behavior_label.npy
dev_user_cache=../../data/large_dev/user.npy
dev_behavior_data_cache=../../data/large_dev/behavior_data.npy
dev_behavior_label_cache=../../data/large_dev/behavior_label.npy
test_user_cache=../../data/large_dev/test_user.npy
test_behavior_data_cache=../../data/large_dev/test_behavior_data.npy
test_behavior_label_cache=../../data/large_dev/test_behavior_label.npy
test_session_cache=../../data/large_dev/session.npy
predict_user_cache=../../data/large_test/user.npy
predict_behavior_data_cache=../../data/large_test/behavior_data.npy
predict_behavior_label_cache=../../data/large_test/behavior_label.npy
predict_session_cache=../../data/large_test/session.npy
predict_impression_id_cache=../../data/large_test/impression_id.npy
all_news_title_cache=../../data/large_train/news_title.npy
news2id_cache=../../data/large_train/new2id.pickle
user2id_cache=../../data/large_train/user2id.pickle
test=./model_best.pth.tar
predict_file=./model_best.pth.tar
epoch=30
version=large
lr=1e-4

CUDA_VISIBLE_DEVICES=4 python3 ../../code/main.py \
--train_news_data ${train_news_data} \
--dev_news_data ${dev_news_data} \
--test_news_data ${test_news_data} \
--train_behavior_data ${train_behavior_data} \
--dev_behavior_data ${dev_behavior_data} \
--test_behavior_data ${test_behavior_data} \
--version ${version} \
--lr ${lr} \
--epoch ${epoch} \
--predict_file ${predict_file} \
--train_user_cache ${train_user_cache} \
--train_behavior_data_cache ${train_behavior_data_cache} \
--train_behavior_label_cache ${train_behavior_label_cache} \
--dev_user_cache ${dev_user_cache} \
--dev_behavior_data_cache ${dev_behavior_data_cache} \
--dev_behavior_label_cache ${dev_behavior_label_cache} \
--test_user_cache ${test_user_cache} \
--test_behavior_data_cache ${test_behavior_data_cache} \
--test_behavior_label_cache ${test_behavior_label_cache} \
--test_session_cache ${test_session_cache} \
--predict_user_cache ${predict_user_cache} \
--predict_behavior_data_cache ${predict_behavior_data_cache} \
--predict_behavior_label_cache ${predict_behavior_label_cache} \
--predict_session_cache ${predict_session_cache} \
--predict_impression_id_cache ${predict_impression_id_cache} \
--all_news_title_cache ${all_news_title_cache} \
--news2id_cache ${news2id_cache} \
--user2id_cache ${user2id_cache}

#--test ${test}