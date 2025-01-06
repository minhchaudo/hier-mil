from utils import get_data, train
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import torch


def extract_cell_type_logits(df, args, perm=False, seed=None, reduce=True):
    samples = df[["patient", "label"]].drop_duplicates()
    if perm:
        samples["label"] = samples["label"].sample(len(samples), random_state=seed).to_list()
    skf = StratifiedKFold(args.n_folds, shuffle=True, random_state=0)
    ct_logits = []
    truths = []
    for train_idx, test_idx in skf.split(samples, samples["label"]):

        train_samples = samples.iloc[train_idx, :]
        test_samples = samples.iloc[test_idx, :]

        X_test, y_test, batch_test, _ = get_data(df, args.all_ct, test_samples, binary=args.binary)

        X_train, y_train, batch_train, _  = get_data(df, args.all_ct, train_samples, binary=args.binary)

        model = train(X_train, y_train, batch_train, None, args, dropout=args.dropout, n_layers_lin=args.n_layers_lin, n_layers_lin2=1, n_layers_lin_meta=args.n_layers_lin_meta, n_hid=args.n_hid, n_hid2=0, lr=args.lr, weight_decay=args.weight_decay, n_epochs=args.n_epochs, seed=0, save=False)

        X_test, y_test, batch_test = X_test.to(args.device), y_test.to(args.device), batch_test.to(args.device)

        with torch.no_grad():
            model.eval()
            ct_logit, _ = model.decompose_logits(X_test, batch_test, len(args.all_ct)*len(y_test), len(args.all_ct))
            ct_logits.extend(ct_logit)
            truths.extend(y_test.long().squeeze().cpu().numpy())
    ct_logits = torch.stack(ct_logits).cpu().numpy()
    truths = np.array(truths)
    if args.binary:
        tmp = pd.DataFrame(ct_logits, columns=args.all_ct)
        tmp["label"] = truths
        tmp = tmp.groupby("label").mean().loc[[0,1],:].to_numpy()
        if reduce:
            tmp = tmp[1] - tmp[0]
    else:
        tmp = []
        for i in range(args.n_classes):
            curr_class_logits = ct_logits[:, :, i]
            other_class_logits = ct_logits[:, :, [c for c in range(args.n_classes) if c != i]].sum(-1)
            diff = curr_class_logits - other_class_logits
            tmp.append(np.mean(diff[truths == i], 0))
        tmp = pd.DataFrame(tmp, columns=args.all_ct).to_numpy()
        if reduce:
            tmp = tmp.sum(0)
    return tmp


def get_p_val_cell_type(df, args):
    orig = extract_cell_type_logits(df, args, perm=False, seed=0, reduce=False)
    orig_importance_score = orig[1] - orig[0]
    orig = pd.DataFrame(orig, columns=args.all_ct)
    perm_importance_scores = []
    for i in range(args.n_perm):
        perm_importance_scores.append(extract_cell_type_logits(df, args, perm=True, seed=i, reduce=True))
    perm_importance_scores = np.stack(perm_importance_scores)
    p_vals = (orig_importance_score[None, :] < perm_importance_scores).sum(0) / args.n_perm
    _, p_vals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
    res = orig.T
    res["importance_score"] = orig_importance_score
    res["adjusted_p_vals"] = p_vals_corrected
    res = res.sort_values("importance_score", ascending=False)
    res.to_csv(args.output)
    return res





