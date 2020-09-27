import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import pdb

class SimilarityEntropy(nn.Module):
    def __init__(self, coef=1, tau=1):
        super().__init__()

        self.tau = tau
        self.coef = coef

    def forward(self, x):
        sim = self.similarity(x)
        loss = self.coef * self.entropy(sim)
        return loss

    def similarity(self, x):
        # dot
#         sim = torch.matmul(x, x.t()) # (num, num)

        # l2 norm
        S = x.unsqueeze(0)  # 1, num, dim
        Q = x.unsqueeze(1)  # num, 1, dim
        sim = -(torch.pow(S - Q, 2)).sum(2) # num, num
        sim -= torch.eye(sim.shape[0]).to(sim.device) * 1e2
        return sim

    def entropy(self, sim):
        prob = torch.softmax(sim * self.tau, dim=-1)        # (num, num)
        info = -torch.log_softmax(sim * self.tau, dim=-1)   # (num, num)
        entropy = torch.sum(torch.mul(prob, info), dim=-1)  # (num,)
        entropy = entropy.mean()                            # (,)
        return entropy
