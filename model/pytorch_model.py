from torch import nn

class CnnLstm(nn.Module):


    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super().__init__()

        ls=2*out_channels*36

        self.cnn1=nn.Conv2d(in_channels,out_channels,kernel_size,stride)
        self.a_cnn1=nn.ReLU()
        self.cnn2=nn.Conv2d(out_channels,2*out_channels,kernel_size,stride)
        self.a_cnn2=nn.ReLU()

        self.f=nn.Flatten(0,2)
        self.seq=nn.Sequential(
            nn.Linear(ls,2*ls),
            nn.ReLU(),
            nn.Linear(2*ls,2*ls),
            nn.ReLU(),
            nn.Linear(2*ls,1),
            nn.ReLU()
        )

    def forward(self,x):

        temp=self.f(self.cnn2(self.cnn1(x)))
        return self.seq(temp)