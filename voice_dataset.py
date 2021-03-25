import torch
from torch import nn,optim
import numpy as np
from   torch.nn import functional as F
from   torch.utils.data import Dataset, DataLoader
import os,sys,glob
import librosa 
import matplotlib.pyplot as plt
import random,csv
import librosa.display

# (128*32)
def convert_mel(path):
    wave_data,sr = librosa.load(path,sr = None)
    num= len(wave_data)
    
    temp = int(1*sr-num)
    list_abordcast =np.array([0 for i in range(temp)])
    #print(len(wave_data))
    if len(wave_data) < 1*sr:                              #不足1s用0进行补全
        #print(wave_data.shape,list_abordcast.shape)
        wave_data=np.concatenate((wave_data,list_abordcast))
    melspec = librosa.feature.melspectrogram(wave_data,sr,n_fft=1024,hop_length=512, n_mels=128)
    logmelspec = librosa.power_to_db(melspec)
    return logmelspec,sr

class voice(Dataset):
    def __init__(self,root):
        super(voice,self).__init__()                          
        self.root=root
        self.namelabel = {}            

        list_voicename = sorted(os.listdir(os.path.join(root)))
        #进行编号
        for name in list_voicename:                                 #遍历每个文件
            if not os.path.isdir(os.path.join(root,name)):          #如果不是文件夹，跳过
                continue
            self.namelabel[name] = len(self.namelabel.keys())       #添加字典元素 
            #这里很巧妙把序号i换成keys值列表长度，也实现i功能，在添加列表可以借鉴,

        self.voices,self.labels = self.load_csv('voices.csv')       #加载每个文件的位置，类别

    def generate_csv(self,filename):               
        #把图片的路径，标签写入csv中
        if not os.path.exists(os.path.join(self.root,filename)):
            voices = []
            for name in self.namelabel.keys():
                voices  += glob.glob(os.path.join(self.root, name, '*.wav'))    #读取name文件夹所有符合的文件
                voices += glob.glob(os.path.join(self.root, name, '*.mp3'))
            #print(len(images), images)
            random.shuffle(voices)                             #打乱列表顺序，打散图片顺序

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for voc in voices:
                    name = voc.split(os.sep)[-2]               #取所对应文件夹名，即为图片类别，os.sep 返回分隔符
                    label = self.namelabel[name]
                    
                    writer.writerow([voc, label])              #一次写入一行
                print('writen into csv file:', filename)

    def load_csv(self,filename):
        #读取csv文件
        if not os.path.exists(os.path.join(self.root,filename)):
            self.generate_csv(filename)

        voices,label = [],[]

        with open(os.path.join(self.root,filename)) as f: 
            reader = csv.reader(f)
            for row in csv.reader(f):
                voc,lab = row
                lab = int(lab)                #
                voices.append(voc)
                label.append(lab)

        assert len(voices) == len(label)

        return voices,label
    def __len__(self):
        return len(self.voices)

    def __getitem__(self,idx):               #复写__fetitem__方法，把张图片数据进行处理，转化为tensor

        voc,label = self.voices[idx],self.labels[idx]    #读取原始数据，与标签

        voc = convert_mel(voc)[0]                        #转化为二维mel时频图
        voc  =  torch.FloatTensor(voc)                   #转化成网络要求的格式FloatTensor
        voc = voc.unsqueeze(0)                           # 增加1个维度，因为通道数不存在
        label = torch.tensor(label)                      # label 转化成要求的格式
        return voc, label    

'''
python -m visdom.server

在浏览器输入：http://localhost:8097/，即可启动
'''

def main():
    
   
    db = voice('./train')    #数据夹path，resize大小，数据集模式
    print('finish')
    x,y = next(iter(db))

    
    
    loader = DataLoader(db,batch_size=20,shuffle=True,num_workers=8)    #num_works  是进程数，
    i=0
    for x,y in loader:
        i +=1
        #print(x.size())

      
'''
if __name__ == '__main__':
    main()
'''


def test():
    path = './train/bed/0b56bcfe_nohash_0.wav' 
    #path = './train/bird/0a7c2a8d_nohash_0.wav'  
    logmelspec,sr =convert_mel(path)
    print(logmelspec.shape)
    
    # plot mel spectrogram

    plt.figure()
    librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Beat wavform')
    plt.show()



def select_data(path):                                  #剔除掉小于16000的数据
    a = os.listdir(path)
    for one_type_path in a:                       #打开每一类文件夹
        temp_path = path+'/'+one_type_path
        list_voc = os.listdir(temp_path)
        for voc_path in list_voc:
            judge_time(temp_path+'/'+voc_path)


def judge_time(path):
    wave_data,sr = librosa.load(path,sr = None)
    num= len(wave_data)
    
    temp = int(1*sr-num)
    
    #print(type(wave_data))
    if len(wave_data) < 1*sr:          #不足1
        #print(wave_data.shape,list_abordcast.shape)
        os.remove(path)

#select_data('./train')             #剔除训练集16000以下声音