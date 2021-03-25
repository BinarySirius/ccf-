import torch
from torch import nn,optim
from resnet import Resnet_pic
import os,sys
import numpy as np
from   torch.nn import functional as F
from voice_dataset import *


def main():
    db = voice('./train')
    loader = DataLoader(db,batch_size=30,shuffle=True,num_workers=8)

    device = torch.device('cuda')
    model = Resnet_pic(num_class=30).to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-4)
    print(model)

    for epoch in range(50):
        acc = 0
        for x,label in loader:
            
            #print(x.shape,len(label))                                        #enumerate 函数封装一层序号
            x,label = x.to(device),label.to(device)
            logits = model(x)
            #print(logits.size())
            #print('log:',logits[0])
            #print(F.softmax(logits[0],dim=0))
            loss = criteon(logits,label)
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            temp = torch.max(logits,1)[1]       #每行的最大值   ,括号的参数是1是求每行的最大值
            '''
            print('pret',temp)  
            print('log',label)
            print('**********************')
            '''     
            acc += ((temp == label).sum().item())/len(label)/len(loader)        #将预测值与真实值比较，求正确的数值比例

            #y_label = np.array(label)
        
        #print('log',label)  
        print(epoch, 'loss:', loss.item(),'acc:',acc)
        
        torch.save(model.state_dict(), './res_params'+str(epoch)+'.pkl')
    '''
    model.eval()

    img=cv2.imread('D:/py/deeplearn/.vscode/xiangmu/data3/face_pic18.jpg')
    img = oneimg_tran(img)

    img= img.to(device)
    out = model(img)
    #out = torch.max(out,1)[1]
    print(F.softmax(out))
    print(out)
    out = torch.max(out,1)[1]
    print(out)
    '''


if __name__ == '__main__':
    main()
