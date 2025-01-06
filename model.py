from torch_geometric.utils import softmax
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch

class Model(torch.nn.Module):
  def __init__(self, n_in, n_out=1, n_in_meta=0, attn1=True, attn2=True, dropout=0.0, use_softmax=True, n_layers_lin=1, n_layers_lin2=0, n_layers_lin_meta=1, n_hid=32, n_hid2=32):
    super().__init__()
    self.lin = torch.nn.Sequential(
        *self.get_lin_layers(n_layers_lin, n_in, n_hid, n_hid, dropout)
    )
    curr_in = n_in if len(self.lin)==0 else n_hid
    self.w_c = torch.nn.Sequential(
        torch.nn.Linear(curr_in, 1),
        torch.nn.Dropout(dropout)
    )
    self.n_in1 = curr_in
    self.lin2 = torch.nn.Sequential(
        *self.get_lin_layers(n_layers_lin2, curr_in, n_hid2, n_hid2, dropout)
    )
    curr_in = curr_in if len(self.lin2)==0 else n_hid2
    self.w_ct = torch.nn.Sequential(
        torch.nn.Linear(curr_in, 1),
        torch.nn.Dropout(dropout)
    )
    if n_in_meta > 0:
        self.lin_meta = torch.nn.Sequential(
            *self.get_lin_layers(n_layers_lin_meta, n_in_meta, curr_in, curr_in, dropout)
        )
        curr_in += (n_in_meta if n_layers_lin_meta == 0 else curr_in)
    self.lin_out = torch.nn.Linear(curr_in, n_out)
    self.attn1 = attn1
    self.attn2 = attn2
    self.use_softmax = use_softmax

  def get_lin_layers(self, n_layers, n_in, n_hid, n_out, dropout):
    layers = []
    for i in range(n_layers):
      curr_in = n_in if i == 0 else n_hid
      curr_out = n_out if i == n_layers - 1 else n_hid
      layers.extend([torch.nn.Linear(curr_in, curr_out), torch.nn.ReLU(), torch.nn.Dropout(dropout)])
    return layers

  def forward(self, X, batch, ct_size, n_ct, meta=None):
    X = self.lin(X)
    if self.attn1:
        if self.use_softmax:
            w_c = softmax(self.w_c(X).squeeze(), batch)
        else:
            w_c = torch.sigmoid((self.w_c(X)).squeeze())
        if self.attn2:
            X = global_add_pool(X * w_c.unsqueeze(dim=-1), batch, size=ct_size).reshape(-1, n_ct, self.n_in1)
        else:
            X = global_add_pool(X * w_c.unsqueeze(dim=-1), batch)
    else:
        if self.attn2:
            X = global_mean_pool(X, batch, size=ct_size).reshape(-1, n_ct, self.n_in1)
        else:
            X = global_mean_pool(X, batch)
    X = self.lin2(X)
    if self.attn2:
        if self.use_softmax:
            w_ct = torch.nn.Softmax(dim=1)(self.w_ct(X))
        else:
            w_ct = torch.sigmoid(self.w_ct(X))
        X = torch.sum(X * w_ct, dim=1)
    if meta is not None:
        meta = self.lin_meta(meta)
        X = torch.cat([X, meta], dim=1)
    X = self.lin_out(X)
    return X

  def decompose_logits(self, X, batch, ct_size, n_ct):
    X = self.lin(X)
    w_c = softmax(self.w_c(X).squeeze(), batch)
    X = global_add_pool(X * w_c.unsqueeze(dim=-1), batch, size=ct_size).reshape(-1, n_ct, self.n_in1)
    X = self.lin2(X)
    w_ct = torch.nn.Softmax(dim=1)(self.w_ct(X))
    X = X @ self.lin_out.weight.T
    return (w_ct * X).squeeze(), w_ct.squeeze()


    
    
