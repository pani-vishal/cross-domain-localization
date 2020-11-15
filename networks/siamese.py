import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-1])
        self.fc1 = nn.Linear(512, 512)
        

    def forward(self, img):
        out = self.resnet(img)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        return out

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    """Code from: https://github.com/adambielski/siamese-triplet"""
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)