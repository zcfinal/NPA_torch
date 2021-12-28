import torch
import transformers
import numpy as np
import ipdb




class NPA(torch.nn.Module):
    def __init__(self,news_input,user2id,word2id,word_embed_pretrain,
                 negtive_ratio=4,
                 max_news_length=30,
                 max_history=50,
                 word_embedding_size=300,
                 cnn_filter_number=400,
                 cnn_window_size=3,
                 user_embedding_size=50,
                 word_preference_size=200,
                 title_preference_size=200):
        super().__init__()
        self.max_news_length = max_news_length
        self.max_history = max_history
        self.negtive_ratio = negtive_ratio
        self.word_embedding_size = word_embedding_size
        self.cnn_window_size = cnn_window_size
        self.cnn_filter_number = cnn_filter_number
        self.user_embedding_size = user_embedding_size
        self.word_preference_size = word_preference_size
        self.title_preference_size = title_preference_size
        self.word_embedding = torch.nn.Embedding(len(word2id)+1,self.word_embedding_size)
        self.word_embedding.weight.data.copy_(torch.from_numpy(word_embed_pretrain))
        self.cnn = torch.nn.Conv1d(in_channels=self.word_embedding_size,
                                   out_channels=self.cnn_filter_number,
                                   kernel_size=self.cnn_window_size,
                                   padding=1)
        self.user_embedding = torch.nn.Embedding(len(user2id),self.user_embedding_size)
        self.word_preference_linear = torch.nn.Linear(in_features=self.user_embedding_size,
                                                      out_features=self.word_preference_size)
        self.title_preference_linear = torch.nn.Linear(in_features=self.user_embedding_size,
                                                       out_features=self.title_preference_size)
        self.word_preference_transform = torch.nn.Linear(in_features=self.word_preference_size,
                                                         out_features=self.cnn_filter_number)
        self.title_preference_transform = torch.nn.Linear(in_features=self.title_preference_size,
                                                          out_features=self.cnn_filter_number)

        self.news_input = torch.from_numpy(news_input).cuda()
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(0.2)

    '''
    input size:[batch_size,news_size]
    user_id size:[batch_size,1]
    
    '''
    def news_encoder(self,input,user_id):
        news_size = input.shape[1]
        # [batch_size,news_size,max_input_length]
        input = self.news_input[input]
        # [batch_size,news_size,max_input_length,word_embedding_size]
        word_embedding_sequence = self.word_embedding(input)
        word_embedding_sequence = self.dropout(word_embedding_sequence)
        # [batch_size*news_size,max_input_length,word_embedding_size]
        word_embedding_sequence = word_embedding_sequence.reshape((-1,self.max_news_length,self.word_embedding_size))
        # [batch_size*news_size,word_embedding_size,max_input_length]
        word_embedding_sequence = torch.transpose(word_embedding_sequence,1,2)
        # [batch_size*news_size,max_input_length,cnn_filter_size]
        word_context_sequence = self.cnn(word_embedding_sequence).transpose(1,2)
        word_context_sequence = self.dropout(word_context_sequence)

        # [batch_size,1,user_embedding_size]
        user_embeddings = self.user_embedding(user_id)
        # [batch_size,1,word_preference_size]
        user_word_preference = self.relu(self.word_preference_linear(user_embeddings))
        # [batch_size,1,cnn_filter_size]
        user_word_preference = self.tanh(self.word_preference_transform(user_word_preference))
        user_word_preference = self.dropout(user_word_preference)
        # [batch_size,cnn_filter_size,1]
        user_word_preference = torch.transpose(user_word_preference,1,2)
        # [batch_size,cnn_filter_size*news_size,1]
        user_word_preference = user_word_preference.repeat(1,news_size,1)
        # [batch_size*news_size,cnn_filter_size,1]
        user_word_preference = torch.reshape(user_word_preference,(-1,self.cnn_filter_number,1))

        # [batch_size*news_size,max_input_length,1]
        attention_score = torch.bmm(word_context_sequence,user_word_preference)
        # [batch_size*news_size,max_input_length]
        attention_score = torch.squeeze(attention_score,dim=2)
        attention_score = torch.nn.functional.softmax(attention_score,dim=1)
        # [batch_size*news_size,1,max_input_length]
        attention_score = torch.unsqueeze(attention_score,dim=1)
        # [batch_size*news_size,1,cnn_filter_size]
        news_representation = torch.bmm(attention_score,word_context_sequence)
        # [batch_size,news_size,1,cnn_filter_size]
        news_representation = torch.reshape(news_representation,(-1,news_size,1,self.cnn_filter_number))
        # [batch_size,news_size,cnn_filter_size]
        news_representation = torch.squeeze(news_representation,2)
        news_representation = self.dropout(news_representation)
        return news_representation

    '''
    history size:[batch_size,history_limit,cnn_filer_size]
    user_id size:[batch_size,1]
    
    '''
    def user_encoder(self,history,user_id):
        # [batch_size,history_limit,cnn_filter_size]
        news_representation = history

        # [batch_size,1,user_embedding_size]
        user_embeddings = self.user_embedding(user_id)
        # [batch_size,1,title_preference_size]
        user_title_preference = self.relu(self.title_preference_linear(user_embeddings))
        # [batch_size,1,cnn_filter_size]
        user_title_preference = self.tanh(self.title_preference_transform(user_title_preference))
        user_title_preference = self.dropout(user_title_preference)
        # [batch_size,cnn_filter_size,1]
        user_title_preference = torch.transpose(user_title_preference,1,2)

        # [batch_size,history_limit,1]
        attention_score = torch.bmm(news_representation,user_title_preference)
        # [batch_size,history_limit]
        attention_score = torch.squeeze(attention_score,dim=2)
        attention_score = torch.nn.functional.softmax(attention_score,dim=1)
        # [batch_size,1,history_limit]
        attention_score = torch.unsqueeze(attention_score,dim=1)
        # [batch_size,1,cnn_filter_size]
        user_representation = torch.bmm(attention_score,news_representation )
        # [batch_size,cnn_filter_size]
        user_representation = torch.squeeze(user_representation,1)
        user_representation = self.dropout(user_representation)
        return user_representation

    '''
    news_rep size = [batch_size,5(1 positive+4 negtive),cnn_filter_size]
    user_rep size = [batch_size,cnn_filter_size]
    '''
    def score_predict(self,candidate_news_representation,user_representation):
        #[batch_size,5]
        score = torch.bmm(candidate_news_representation,torch.unsqueeze(user_representation,2)).squeeze(2)
        return score

    '''
    input size:[batch_size,5(1 positive+4 negtive) + 50(history)]
    user_id: [batch_size,1]
    '''
    def forward(self,input,user_id):
        news_rep = self.news_encoder(input,user_id)
        user_rep = self.user_encoder(news_rep[:,1+self.negtive_ratio:,:],user_id)
        score = self.score_predict(news_rep[:,:1+self.negtive_ratio,:],user_rep)
        return score
