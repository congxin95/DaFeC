import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import pdb

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return x

class Relation(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=230, dropout=0.1):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(dropout)
#         self.score = nn.Sequential(
#                 nn.Conv1d(4, 3, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool1d(2),
#                 nn.Conv1d(3, 5, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool1d(2),
#                 Flatten(),
#                 nn.Linear(285, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 1)
#                 )
        
#         self.score = nn.Sequential(
#                 nn.Linear(hidden_size*2, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 1)
#                 )

    def similarity(self, S, Q):
        # l2 norm
        S = S.unsqueeze(1)
        Q = Q.unsqueeze(2)
        sim = -(torch.pow(S - Q, 2)).sum(3)

        # relation
#         _, N, dim = S.shape
#         _, NQ, dim = Q.shape
#         S = S.unsqueeze(1)
#         Q = Q.unsqueeze(2)
#         S = S.expand(-1, NQ, N, dim)
#         Q = Q.expand(-1, NQ, N, dim)
        
#         pdb.set_trace()
#         feature = torch.cat([S, Q], dim=-1)
#         sim = self.score(feature).squeeze(-1)
        
#         S = S.contiguous().view(-1, dim)
#         Q = Q.contiguous().view(-1, dim)
#         S_minus_Q = torch.abs(S - Q)
#         S_mul_Q = torch.mul(S, Q)
#         feature = torch.stack([S, Q, S_minus_Q, S_mul_Q], dim=-2)
#         sim = self.score(feature)
#         sim = sim.view(-1, NQ, N)

        return sim

    def proto(self, support):
        prototype = torch.mean(support, 2)
        return prototype

    def forward(self, support, query, N, K, total_Q):
        support_emb = self.sentence_encoder(support)
        query_emb = self.sentence_encoder(query)
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        support = support.view(-1, N, K, self.hidden_size)
        query = query.view(-1, total_Q, self.hidden_size)

        # Prototypical Networks 
        prototype = self.proto(support)
        logits = self.similarity(prototype, query)
        _, pred = torch.max(logits.view(-1, N), 1)
        
        return logits, pred
