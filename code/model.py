import torch
import transformers
import numpy as np
import ipdb





class Amm_title(torch.nn.Module):
    def __init__(self,hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.bert_model = transformers.AutoModel.from_pretrained("prajjwal1/bert-mini")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size)
        self.fc_1 = torch.nn.Linear(in_features=self.hidden_size,out_features=self.hidden_size)
        self.fc_2 = torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
    '''
    input size:[batch_size,max_input_length]
    
    
    '''
    def forward(self,input):
        # [batch_size,max_input_length,hidden_size]
        hidden_state = self.bert_model(**input,)[0]
        cls_tensor = hidden_state[:,0,:]
        # [batch_siz,hidden_size]
        cls_tensor = torch.squeeze(cls_tensor,1)
        residual = cls_tensor
        output = self.fc_2(self.fc_1(cls_tensor))
        output = self.layer_norm(output + residual)
        return output


