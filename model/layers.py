"""
Additional layers.
"""
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

from utils import constant, torch_utils

class LSTMLayer(nn.Module):
    """ A wrapper for LSTM with sequence packing. """

    def __init__(self, emb_dim, hidden_dim, num_layers, dropout, use_cuda):
        super(LSTMLayer, self).__init__()
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.use_cuda = use_cuda

    def forward(self, x, x_mask, init_state):
        """
        x: batch_size * feature_size * seq_len
        x_mask : batch_size * seq_len
        """
        x_lens = x_mask.data.eq(constant.PAD_ID).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lens = list(x_lens[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)
        
        # sort by seq lens
        x = x.index_select(0, idx_sort)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)
        rnn_output, (ht, ct) = self.rnn(rnn_input, init_state)
        rnn_output = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)[0]
        
        # unsort
        rnn_output = rnn_output.index_select(0, idx_unsort)
        ht = ht.index_select(0, idx_unsort)
        ct = ct.index_select(0, idx_unsort)
        return rnn_output, (ht, ct)

class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """
    
    def __init__(self, input_size, query_size, feature_size, attn_size):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
        else:
            self.wlinear = None
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.vlinear.weight.data.normal_(std=0.001)
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_() # use zero to give uniform attention at the beginning
    
    def forward(self, x, x_mask, q, f):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)
        q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
            batch_size, self.attn_size).unsqueeze(1).expand(
                batch_size, seq_len, self.attn_size)
        if self.wlinear is not None:
            f_proj = self.wlinear(f.view(-1, self.feature_size)).contiguous().view(
                batch_size, seq_len, self.attn_size)
            projs = [x_proj, q_proj, f_proj]
        else:
            projs = [x_proj, q_proj]
        scores = self.tlinear(F.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len)

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores)
        # weighted average input vectors
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
        return outputs

class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        # xWy.data.masked_fill_(x_mask.data, -float('inf'))
        # if self.training:
        #     # In training we output log-softmax for NLL
        #     alpha = F.log_softmax(xWy, dim=1)
        # else:
        #     # ...Otherwise 0-1 probabilities
        #     alpha = F.softmax(xWy, dim=1)
        return xWy


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        # scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)
        return alpha

# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked input."""
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)
