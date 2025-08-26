from torch import nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import torch
class fcmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(9024, 256)
        # self.hidden1 = nn.Linear(336, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 256)
        self.hidden4 = nn.Linear(256, 256)
        self.hidden5 = nn.Linear(256, 256)
        self.hidden6 = nn.Linear(256, 256)
        self.hidden7 = nn.Linear(256, 256)
        self.hidden8 = nn.Linear(256, 256)
        self.hidden9 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 128)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.reshape(-1, 9024)
        # x = x.reshape(-1, 336)
        x = x.float()
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
        x = F.relu(self.hidden9(x))
        x = self.out(x)
        return x

class Network(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = fcmodel()
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c