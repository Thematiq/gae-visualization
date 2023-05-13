import torch.nn.functional as F

from torch.nn import Module
from torch_geometric.nn import GCNConv, GATv2Conv


class GCNEncoder(Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)  # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)  # cached only for transductive learning

    def forward(self, x, edge_index, _e_a=None):
        x = self.conv1(x, edge_index).tanh()
        return self.conv2(x, edge_index)


class GCNDecoder(Module):
    def __init__(self, in_channels, out_channels):
        super(GCNDecoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)  # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)  # cached only for transductive learning

    def forward(self, x, edge_index, _e_a=None):
        x = self.conv1(x, edge_index).tanh()
        return self.conv2(x, edge_index)


class AttentionEncoder(Module):
    def __init__(self, in_channels, out_channels, attention_heads=1):
        super().__init__()
        mid = (in_channels + out_channels) // 2
        self.conv1 = GATv2Conv(in_channels, mid, edge_dim=1, attention_heads=attention_heads,
                               cached=True)
        self.conv2 = GATv2Conv(mid, out_channels, edge_dim=1, attention_heads=attention_heads,
                               cached=True)

    def forward(self, x, e_i, e_a):
        x = self.conv1(x, e_i, e_a)
        x = F.gelu(x)
        x = self.conv2(x, e_i, e_a)
        return x


class AttentionDecoder(Module):
    def __init__(self, in_channels, out_channels, attention_heads=1):
        super().__init__()
        mid = (in_channels + out_channels) // 2
        self.conv1 = GATv2Conv(in_channels, mid, edge_dim=1, attention_heads=attention_heads,
                               cached=True)
        self.conv2 = GATv2Conv(mid, out_channels, edge_dim=1, attention_heads=attention_heads,
                               cached=True)

    def forward(self, x, e_i, e_a):
        x = self.conv1(x, e_i, e_a)
        x = F.gelu(x)
        x = self.conv2(x, e_i, e_a)
        return x
