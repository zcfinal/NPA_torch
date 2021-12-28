import torch
import numpy as np

def predict_result(model,test_data_loader,test_data):
    model.eval()
    predict_score = []
    for i, (input, user) in enumerate(test_data_loader):
        input = input.cuda(non_blocking=True)
        user = user.cuda(non_blocking=True)
        #[batch,1]
        output = model(input,user)
        predict_score.append(output.cpu().detach())
        if i%10000==0:
            print('precit cases:{}/{}'.format(i,len(test_data_loader)))

    predict_score = torch.cat(predict_score,0)
    predict_score = predict_score.numpy()

    ans = []
    for session in test_data.session:
        session_score = predict_score[session[0]:session[1],0]
        order = np.argsort(session_score)[::-1].tolist()
        session_ans = [0]*(session[1]-session[0])
        for i, index in enumerate(order,start=1):
            session_ans[index]=i
        ans.append(session_ans)

    impression_id = test_data.impression_id

    with open('prediction.txt','w') as fout:
        for imp_id,session_ans in zip(impression_id,ans):
            fout.write(str(imp_id)+' [')
            session_ans = [str(rank) for rank in session_ans]
            fout.write(','.join(session_ans)+']\n')

