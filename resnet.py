import  torch
from    torch import  nn
from    torch.nn import functional as F

#(128*32)   30类
class Resblk(nn.Module):                                 #建立残差网络，
    '''
        残差块由两个3*3卷积，短接层由1*1的卷积层组成
    '''
    def __init__(self,channal_in,channal_out,stride = 1):
        super(Resblk,self).__init__()

        self.conv1 = nn.Conv2d(channal_in,channal_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(channal_out)
        self.conv2 = nn.Conv2d(channal_out,channal_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(channal_out)

        self.extra = nn.Sequential()            #  保证短接层输出通道数与卷积出的通道数一样           
        if channal_in != channal_out:                                       
            self.extra = nn.Sequential(
                            nn.Conv2d(channal_in,channal_out,kernel_size=1,stride=stride),
                            nn.BatchNorm2d(channal_out)
                                     )
    
        
    def forward(self,x):
        orginal_data = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.extra(orginal_data) + x
        return x

class Resnet_pic(nn.Module):                   #输入为128*32像素,传入要分类个数
    def __init__(self,num_class=30):
        super(Resnet_pic,self).__init__()
         #[b,64,w,h]>[b,64,w/2,h/2]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride = 1,padding = 0),
            nn.BatchNorm2d(32)
            )

        #follow 4 blocks 
        #[b,64,w/2,h/2]>[b,64,w/4,h/4]
        self.bk1= Resblk(32,64,stride=2)   
        #[b,128,w,h]
        self.bk2 = Resblk(64,128,stride=2)
        #[b,128,w/16,h/16]
        self.bk3 = Resblk(128,256,stride=2)
        #[b,128,w,h]
        #self.bk4 = Resblk(256,512,stride=2)
        


        self.fc_unit = nn.Sequential(
            nn.Linear(16*4*256,3000),
            nn.ReLU(),
            nn.Linear(3000,600),
            nn.ReLU(),
            nn.Linear(600,120),
            nn.ReLU(),
            nn.Linear(120,num_class)
        )
        
        x = torch.randn(50,1,128,32)
        x=  self.conv1(x)     
        x = self.bk1(x)
        x = self.bk2(x)
        x = self.bk3(x)
        #x=  self.bk4(x)
        print(x.shape)
        
        
    def forward(self,x):
        x= self.conv1(x)

        x = F.relu(x)
        x = self.bk1(x)
        x = self.bk2(x)
        x = self.bk3(x)
        #x = self.bk4(x)
        #x = F.adaptive_avg_pool2d(x, [1, 1])
        
        x = x.view(x.size(0), -1)
        x = self.fc_unit(x)
        return x



def main():
    a= Resnet_pic(num_class=30)
    x = torch.randn(50,1,128,32)
    out = a(x)
    print('resnet:', out.shape)



if __name__ == "__main__":
    main()
        



        
