import torch
import torch.nn as nn
import torch.nn.functional as F

class Disstill(nn.Module):
    def __init__(self,act = nn.ReLU(True)):
        super(Disstill, self).__init__()

        above = [nn.Conv2d(64,48,kernel_size=3,padding=1),act,
                 nn.Conv2d(48,32,kernel_size=3,padding=1),act,
                 nn.Conv2d(32,64,kernel_size=3,padding=1),act]
        below = [nn.Conv2d(48,64,kernel_size=3,padding=1),act,
                 nn.Conv2d(64,48,kernel_size=3,padding=1),act,
                 nn.Conv2d(48,80,kernel_size=3,padding=1),act]
        final_conv = [nn.Conv2d(80,64,kernel_size=1),act]
        self.layers1 = nn.Sequential(*above)
        self.layers2 = nn.Sequential(*below)
        self.layers3 = nn.Sequential(*final_conv)

    def forward(self, x):
        tmp = self.layers1(x)
        tmp ,res = torch.split(tmp,(48,16),dim=1)
        tmp = self.layers2(tmp)
        x = torch.cat((x,res),dim=1)
		output = self.layers3(x + tmp)
		return output
	
class IDN(nn.Module):
	def __init__(self,n_DBlock,act = nn.ReLU(True)):
		super(IDN,self).__init__()
		self.n_DBlock = n_DBlock
		model = [nn.Conv2d(3,64,kernel_size=3,padding=1),act]
		for i in range(n_DBlock):
			model += [Disstill()]
		model += [nn.Conv2d(64,32,kernel_size=3,padding=1),act,
					nn.Conv2d(32,3,kernel_size=1,padding=0)]
		self.model = nn.Sequential(*model)
	def forward(self,x):
		output = x+self.model(x)
		return output

def test_run():
	device = torch.device('cuda')
	x = torch.ones([1,3,15,15], dtype=torch.float32).to(device)
	model = IDN(3).to(device)
	model(x)
if __name__=='__main__':
	test_run()