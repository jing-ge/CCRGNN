from CCRDataset import CCRDataset
from torch_geometric.data import DataLoader,Batch
from sklearn.metrics import recall_score,accuracy_score,f1_score

from model import CCRGNN
import torch
import torch.nn.functional as F
import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def metric(model,y_true,y_pred):
    recall = recall_score(y_true,y_pred, average='weighted')
    acc = accuracy_score(y_true,y_pred)
    f1_s = f1_score(y_true,y_pred,average='weighted')
    return [model,acc,recall,f1_s]



data = CCRDataset('CCRDataset')
train_dataset = data[:2557]
test_datasetloader = DataLoader(data[2557:], batch_size=2557, shuffle=True)
for i in test_datasetloader:
    test_dataset = i
loader = DataLoader(train_dataset, batch_size=640, shuffle=True)


model = CCRGNN(1,9)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

epochs = 2000
score = []

for i in range(epochs):
    model.eval()
    out = model(test_dataset).cpu()
    lb = torch.argmax(F.softmax(out,1), 1).cpu().numpy()
    testacc = metric("CRCNN",test_dataset.y.numpy(),lb)

    loss_epoch = 0
    model.train()
    for batch in loader:
        out = model(batch)
        loss = F.cross_entropy(out,batch.y.cuda())
        loss_epoch += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    score.append(testacc[1:])
    nscore = np.array(score).max(0)
    print("epoch:{} loss:{}".format(i,loss_epoch))
    print("      test  : recall : {:.7f} , acc : {:.7f} , f1-score : {:.7f}".format(testacc[1],testacc[2],testacc[3]))
    print("      best  : recall : {:.7f} , acc : {:.7f} , f1-score : {:.7f}".format(nscore[0],nscore[1],nscore[2]))

# print(train)
# print(test)
# print(d.num_classes)
# print(d.num_node_features)