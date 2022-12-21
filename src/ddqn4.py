
import torch.nn as nn
import torch.nn.functional as F

class DDQN4(nn.Module):
    # 4 frames,

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        linear_input_size = 7 * 7 * 64
        fc_output_size=512
        self.fc_val=nn.Linear(linear_input_size, fc_output_size)
        self.fc_adv=nn.Linear(linear_input_size, fc_output_size)
        self.val = nn.Linear(fc_output_size, 1)
        self.adv = nn.Linear(fc_output_size, 2)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x_val = F.relu(self.fc_val(x.view(x.size(0), -1)))
        x_adv = F.relu(self.fc_adv(x.view(x.size(0), -1)))
        val=self.val(x_val)
        adv=self.adv(x_adv)
        
        
        x=val+adv-adv.mean(1,keepdim=True)
        return x