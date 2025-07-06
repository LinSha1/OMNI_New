import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .graph import create_e_matrix, normalize_digraph
from .graph_edge_model import GEM
from .basic_block import *


# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        self.U2 = nn.Linear(dim_in, dim_out, bias=False)
        self.V2 = nn.Linear(dim_in, dim_out, bias=False)
        self.A2 = nn.Linear(dim_in, dim_out, bias=False)
        self.B2 = nn.Linear(dim_in, dim_out, bias=False)
        self.E2 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.bnv1 = nn.BatchNorm1d(num_classes)
        self.bne1 = nn.BatchNorm1d(num_classes * num_classes)

        self.bnv2 = nn.BatchNorm1d(num_classes)
        self.bne2 = nn.BatchNorm1d(num_classes * num_classes)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        scale = gain * np.sqrt(2.0 / dim_in)
        for layer in [self.U1, self.V1, self.A1, self.B1, self.E1,
                      self.U2, self.V2, self.A2, self.B2, self.E2]:
            layer.weight.data.normal_(0, scale)
        bn_init(self.bnv1)
        bn_init(self.bne1)
        bn_init(self.bnv2)
        bn_init(self.bne2)

    def forward(self, x, edge):
        dev = x.get_device()
        if dev >= 0:
            start = self.start.to(dev)
            end = self.end.to(dev)

        # Layer 1
        res = x
        Vix = self.A1(x)
        Vjx = self.B1(x)
        e = self.E1(edge)
        edge = edge + self.act(
            self.bne1(
                torch.einsum('ev, bvc -> bec', (end, Vix)) +
                torch.einsum('ev, bvc -> bec', (start, Vjx)) + e
            )
        )
        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)
        Ujx = self.V1(x)
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))
        Uix = self.U1(x)
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes
        x = self.act(res + self.bnv1(x))
        res = x

        # Layer 2
        Vix = self.A2(x)
        Vjx = self.B2(x)
        e = self.E2(edge)
        edge = edge + self.act(
            self.bne2(
                torch.einsum('ev, bvc -> bec', (end, Vix)) +
                torch.einsum('ev, bvc -> bec', (start, Vjx)) + e
            )
        )
        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)
        Ujx = self.V2(x)
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))
        Uix = self.U2(x)
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes
        x = self.act(res + self.bnv2(x))
        return x, edge  # [B, num_node, C], [B, num_edge, C]


class GNN_Node(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=5, metric='dots'):
        super(GNN_Node, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        self.U = nn.Linear(self.in_channels, self.in_channels)
        self.V = nn.Linear(self.in_channels, self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape
        # build dynamic graph
        if self.metric == 'dots':
            sim = torch.einsum('b i j , b j k -> b i k', x.detach(), x.detach().transpose(1, 2))
            thr = sim.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:,:, -1].view(b, n, 1)
            adj = (sim >= thr).float()
        elif self.metric == 'cosine':
            normed = F.normalize(x.detach(), p=2, dim=-1)
            sim = torch.einsum('b i j , b j k -> b i k', normed, normed.transpose(1, 2))
            thr = sim.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:,:, -1].view(b, n, 1)
            adj = (sim >= thr).float()
        elif self.metric == 'l1':
            diff = torch.abs(x.detach().unsqueeze(1) - x.detach().unsqueeze(2))
            sim = diff.sum(dim=-1)
            thr = sim.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:,:, -1].view(b, n, 1)
            adj = (sim <= thr).float()
        else:
            raise ValueError(f"Unsupported metric {self.metric}")

        A = normalize_digraph(adj)
        agg = torch.einsum('b i j, b j k -> b i k', A, self.V(x))
        x = self.relu(x + self.bnv(agg + self.U(x)))
        return x


class Head(nn.Module):
    def __init__(self, in_channels, num_node, num_classes, neighbor_num):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.num_node = num_node
        self.num_classes = num_classes
        self.edge_extractor = GEM(self.in_channels, self.num_node)
        self.gnn = GNN(self.in_channels, self.num_node)
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        # contrastive projection head
        self.proj_head = nn.Sequential(
            nn.Linear(self.in_channels, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward(self, global_features, box_features):
        batch_size = global_features.shape[0]

        print(f"Debug - Global features shape: {global_features.shape}")
        print(f"Debug - Box features shape: {box_features.shape}")

        # Handle the 3D case: [150, 49, 512] -> [batch, proposals, spatial, channels]
        total_proposals, spatial_features, channels = box_features.shape
        num_proposals_per_batch = total_proposals // batch_size

        print(f"Debug - {total_proposals} proposals, {num_proposals_per_batch} per batch, {channels} channels")

        # Reshape to [batch, num_proposals, spatial_features, channels]
        box_features = box_features.view(batch_size, num_proposals_per_batch, spatial_features, channels)

        # For node features: average pool spatial dimension to get [batch, num_proposals, channels]
        f_v = box_features.mean(dim=2)
        print(f"Debug - Node features (f_v) shape: {f_v.shape}")

        # Simple approach: create dummy edge features that match the expected dimensions
        # The GNN expects f_e to have shape [batch, num_edges, channels]
        # For a fully connected graph with num_proposals nodes, we have num_proposals^2 edges
        num_edges = num_proposals_per_batch * num_proposals_per_batch

        # Create edge features by computing pairwise differences or similarities
        # Expand f_v to compute pairwise relationships
        f_v_i = f_v.unsqueeze(2).repeat(1, 1, num_proposals_per_batch, 1)  # [batch, proposals, proposals, channels]
        f_v_j = f_v.unsqueeze(1).repeat(1, num_proposals_per_batch, 1, 1)  # [batch, proposals, proposals, channels]

        # Simple edge features: concatenate or difference
        # Using difference for now (could also use concatenation or dot product)
        edge_features = f_v_i - f_v_j  # [batch, proposals, proposals, channels]
        f_e = edge_features.view(batch_size, num_edges, channels)  # [batch, num_edges, channels]

        print(f"Debug - Edge features (f_e) shape: {f_e.shape}")

        # Apply GNN
        f_v, f_e = self.gnn(f_v, f_e)

        print(f"Debug - After GNN - f_v: {f_v.shape}, f_e: {f_e.shape}")

        # Classification logits
        cl = self.fc(f_v.view(-1, f_v.size(2)))

        # Contrastive embeddings
        node_feats = f_v.view(-1, f_v.size(2))  # [B*num_node, C]
        z = self.proj_head(node_feats)  # [B*num_node, 128]
        z = F.normalize(z, dim=1)

        return cl, z


class MEFARG(nn.Module):
    def __init__(self, num_node=50, num_classes=11, neighbor_num=5):
        super(MEFARG, self).__init__()
        self.linear1 = nn.Linear(1050, 49)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(256, 512)
        self.head = Head(512, num_node, num_classes, neighbor_num)

    def forward(self, x):
        global_features, box_features = x
        global_features = global_features['3']  # P5 level
        b, c, h, w = global_features.shape
        global_features = global_features.view(b, c, -1)
        global_features = self.linear1(global_features).permute(0, 2, 1)
        global_features = self.linear2(global_features)

        b, c, h, w = box_features.shape
        box_features = box_features.view(b, c, -1).permute(0, 2, 1)
        box_features = self.linear3(box_features)

        cl, z = self.head(global_features, box_features)
        return cl, z
