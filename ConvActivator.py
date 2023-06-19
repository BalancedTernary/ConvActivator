class ConvActivator(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5,**kwargs):
        super(ConvActivator, self).__init__()
        self.Conv1=torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,bias=True,**kwargs)
        self.Conv2=torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,bias=False,**kwargs)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self._initialize_weights()

    def forward(self,input):
        output=self.Conv1(input.clamp(min=0))+self.Conv2(input.clamp(max=0))
        return output

    def _initialize_weights(self):
        with torch.no_grad():
            gain=1.75/(math.sqrt(self.in_channels)*self.kernel_size)
            self.Conv1.weight.data.uniform_(-gain, gain)
            self.Conv1.bias.data.zero_()
            self.Conv2.weight.data.uniform_(-gain, gain)
            
