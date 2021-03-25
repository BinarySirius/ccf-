import torch
from torch import nn,optim
from resnet import Resnet_pic
import os,sys
import numpy as np
from torch.nn import functional as F
from voice_dataset import *


def get_labeldict(label_root):
    namelabel = {}
    list_voicename = sorted(os.listdir(label_root))
    for name in list_voicename:
        if not os.path.isdir(label_root):
            continue
        namelabel[name] = len(namelabel.keys()) 
    return namelabel

def generate_csv(filename,results):               
    with open('./'+filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        for result in results:
            name = result[0]               #取所对应文件夹名，即为图片类别，os.sep 返回分隔符
            label = result[1]
                    
            writer.writerow([name, label])              #一次写入一行
        print('writen into csv file:', filename)
    
def convert_mel(path):
    wave_data,sr = librosa.load(path,sr = None)
    num= len(wave_data)
    
    temp = int(1*sr-num)
    list_abordcast =np.array([0 for i in range(temp)])
    #print(type(wave_data))
    if len(wave_data) < 1*sr:          #不足1
        #print(wave_data.shape,list_abordcast.shape)
        wave_data=np.concatenate((wave_data,list_abordcast))
    melspec = librosa.feature.melspectrogram(wave_data,sr,n_fft=1024,hop_length=512, n_mels=64)
    logmelspec = librosa.power_to_db(melspec)
    return logmelspec,sr


def detect_voice(path):
    test_list = os.listdir(path)
    result = [['file_name','label']]
    for voc in test_list:
        voc_path = './test/'+voc
        mel_map  =convert_mel(voc_path)[0]         #读取声音文件 
        '''
        转化为torch 的格式
        ''' 
        mel_map =  torch.FloatTensor(mel_map)
        mel_map = mel_map.unsqueeze(0)            #加一条通道
        mel_map = mel_map.unsqueeze(0)            #加一batchsize

        mel_map = mel_map.to(device)

        out=net(mel_map)
        idx = int(torch.max(out,1)[1].item())          #取最大概率

        label = label_key[idx]                  #字典返回list
        result.append([voc,label])
    generate_csv("submission.csv",result)
        


path = './test'
model_path = './pkl/res_params.pkl'
label_root = './train'

                                         
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Resnet_pic(num_class=30).to(device)     #加载网络

state_dict = torch.load(model_path, map_location=device)
net.load_state_dict(state_dict)                                #加载参数模型
    
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#net = nn.DataParallel(net)
print('Finished!')


namelabel =get_labeldict(label_root)
label_key = list(namelabel.keys())

detect_voice(path)

    


