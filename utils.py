import random
import torch
import numpy as np
import pandas as pd
from model import Model

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def get_df(adata, patient_id_key="patient", label_key="label", cell_type_annot_key="cell_type_annotation", no_label=False):
    try:
        df = pd.DataFrame(adata.X.toarray())
    except:
        df = pd.DataFrame(adata.X)
    df.index = adata.obs.index
    if not no_label:
        df[["patient","cell_type_annotation","label"]] = adata.obs[[patient_id_key,cell_type_annot_key, label_key]]
    else:
        df[["patient","cell_type_annotation"]] = adata.obs[[patient_id_key,cell_type_annot_key]]
        df["label"] = -1
    return df

def get_meta(adata, meta_cols, patient_id_key="patient"):
    return adata.obs[meta_cols+[patient_id_key]].drop_duplicates().set_index(patient_id_key)

def get_data(df, all_ct, samples, meta=None, binary=True, attn2=True):
    ct_dict = dict({ct: idx for idx, ct in enumerate(all_ct)})
    Xs = []
    batches = []

    if meta is not None:
        meta = torch.tensor(meta.loc[samples["patient"].to_list(), :].to_numpy(), dtype=torch.float)

    for idx, sample in enumerate(samples["patient"].to_list()):
        sample_df = df[df["patient"]==sample]
        x = sample_df.iloc[:,:df.shape[-1]-3].to_numpy()
        batch = [(idx * len(all_ct) + ct_dict[ct]) for ct in sample_df["cell_type_annotation"].to_list()]\
                if attn2 else [idx for _ in range(len(sample_df))]
        Xs.append(x)
        batches.append(batch)
    Xs = torch.tensor(np.concatenate(Xs), dtype = torch.float)
    batches = torch.tensor(np.concatenate(batches))
    ys = torch.tensor(samples["label"].to_list(), dtype = torch.float if binary else torch.long)
    return Xs, ys, batches, meta

# In args: attn1, attn2, use_meta, binary, n_classes, device, all_ct, model_save_path, n_skf_in, n_perm

def train(X_train, y_train, batch_train, meta_train, args, dropout=0., n_layers_lin=1, n_layers_lin2=0, n_layers_lin_meta=1, n_hid=32, n_hid2=0, lr=1e-3, weight_decay=0., n_epochs=100, seed=0, save=False):
    X_train, y_train, batch_train, meta_train = X_train.to(args.device), y_train.to(args.device), batch_train.to(args.device), meta_train.to(args.device) if meta_train is not None else meta_train
    set_seeds(seed)
    model = Model(X_train.shape[-1], n_out=1 if args.binary else args.n_classes, n_in_meta=0 if not args.use_meta else meta_train.shape[-1], \
    attn1=args.attn1, attn2=args.attn2, use_softmax=True, dropout=dropout, n_layers_lin=n_layers_lin, n_layers_lin2=0, \
    n_layers_lin_meta=1 if not args.use_meta else n_layers_lin_meta, n_hid=n_hid, n_hid2=0).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss() if args.binary else torch.nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_train, batch_train, len(args.all_ct)*len(y_train), len(args.all_ct), meta=meta_train)
        loss = loss_fn(pred.squeeze(), y_train.squeeze())
        loss.backward()
        opt.step()
    if save:
       torch.save(model, args.model_save_path)
    return model

def load(model_save_path):
    model = torch.load(model_save_path, weights_only=False)
    return model

def predict(model, X_test, batch_test, meta_test, n_samples_test, args):
    X_test, batch_test, meta_test = X_test.to(args.device), batch_test.to(args.device), meta_test.to(args.device) if meta_test is not None else meta_test
    if isinstance(model, str):
        model = load(model).to(args.device)
    with torch.no_grad():
        model.eval()
        pred = model(X_test, batch_test, len(args.all_ct)*n_samples_test, len(args.all_ct), meta=meta_test)
        pred = torch.sigmoid(pred.squeeze()) if args.binary else torch.softmax(pred.squeeze(), -1)
    return pred
