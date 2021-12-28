import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def test(model,test_data_loader,test_data):
    model.eval()
    predict_score = []
    for i, (input, user, target) in enumerate(test_data_loader):
        input = input.cuda(non_blocking=True)
        user = user.cuda(non_blocking=True)
        #[batch,1]
        output = model(input,user)
        output = torch.nn.functional.sigmoid(output)
        predict_score.append(output)
        if i%5000==0:
            print('test cases:{}/{}'.format(i,len(test_data_loader)))

    predict_score = torch.cat(predict_score,0)
    predict_score = predict_score.cpu().detach().numpy()
    label = test_data.label.numpy()
    all_auc=[]
    all_mrr=[]
    all_ndcg=[]
    all_ndcg2=[]
    for session in test_data.session:
        if np.sum(label[session[0]:session[1]])!=0 and session[1]<predict_score.shape[0]:
            all_auc.append(roc_auc_score(label[session[0]:session[1]],predict_score[session[0]:session[1],0]))
            all_mrr.append(mrr_score(label[session[0]:session[1]],predict_score[session[0]:session[1],0]))
            all_ndcg.append(ndcg_score(label[session[0]:session[1]],predict_score[session[0]:session[1],0],k=5))
            all_ndcg2.append(ndcg_score(label[session[0]:session[1]],predict_score[session[0]:session[1],0],k=10))

    with open('result.txt','w')as fout:
        fout.write('auc:')
        fout.write(str(np.mean(all_auc)))
        fout.write('\nmrr:')
        fout.write(str(np.mean(all_mrr)))
        fout.write('\nndcg5:')
        fout.write(str(np.mean(all_ndcg)))
        fout.write('\nndcg10:')
        fout.write(str(np.mean(all_ndcg2)))

