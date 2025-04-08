from utils import get_data, set_seeds, train, predict
from sklearn.model_selection import StratifiedKFold
from model import Model
import torch
from sklearn.metrics import roc_auc_score
import itertools
import numpy as np
import pandas as pd

import optuna as optuna
from optuna.samplers import TPESampler

# args: n_repeats, n_folds, n_folds_hyperparam_tune

def objective_wrapper(df, meta, train_samples, args):
    def objective(trial):
        n_epochs = trial.suggest_categorical("n_epochs", [100, 500, 1000])
        dropout = trial.suggest_categorical("dropout", [0, 0.3, 0.5, 0.7])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2])
        n_layers_lin = trial.suggest_categorical("n_layers_lin", [1, 2])
        n_hid = trial.suggest_categorical("n_hid", [32, 64, 128])
        lr = trial.suggest_categorical("lr", [1e-3, 5e-3])
        n_layers_lin_meta = trial.suggest_categorical("n_layers_lin_meta", [0, 1, 2]) if args.use_meta else 1

        skf = StratifiedKFold(args.n_folds_hyperparam_tune, shuffle=True, random_state=0)
        preds_ = []
        truths_ = []

        for train_idx, val_idx in skf.split(train_samples, train_samples["label"]):
            train_samples_in = train_samples.iloc[train_idx, :]
            val_samples = train_samples.iloc[val_idx, :]

            X_train, y_train, batch_train, meta_train = get_data(df, args.all_ct, train_samples_in, binary=args.binary, meta=meta if args.use_meta else None, attn2=args.attn2)
            X_val, y_val, batch_val, meta_val = get_data(df, args.all_ct, val_samples, binary=args.binary, meta=meta if args.use_meta else None, attn2=args.attn2)

            model = train(X_train, y_train, batch_train, meta_train, args, dropout=dropout, n_layers_lin=n_layers_lin, n_layers_lin_meta=n_layers_lin_meta, n_hid=n_hid, lr=lr, weight_decay=weight_decay, n_epochs=n_epochs, seed=0)

            pred = predict(model, X_val, batch_val, meta_val, len(y_val), args).cpu().numpy()
            preds_.extend(pred)
            truths_.extend(y_val)
        return roc_auc_score(np.stack(truths_), np.stack(preds_), multi_class="ovo")
    return objective

def train_and_tune(df, meta, args):
    samples = df[["patient", "label"]].drop_duplicates()

    sampler = TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_wrapper(df, meta, samples, args), n_trials=args.n_tune_trials)
    best_params = study.best_params

    X, y, batch, meta  = get_data(df, args.all_ct, samples, binary=args.binary, meta=meta if args.use_meta else None, attn2=args.attn2)

    model = train(X, y, batch, meta, args, dropout=best_params["dropout"], n_layers_lin=best_params["n_layers_lin"], n_layers_lin_meta=1 if not args.use_meta else best_params["n_layers_lin_meta"], n_hid=best_params["n_hid"], lr=best_params["lr"], weight_decay=best_params["weight_decay"], n_epochs=best_params["n_epochs"], save=True)

    return model

def predict_and_save(df, meta, args):
    samples = df[["patient", "label"]].drop_duplicates()
    X, _, batch, meta  = get_data(df, args.all_ct, samples, binary=args.binary, meta=meta if args.use_meta else None, attn2=args.attn2)
    pred = predict(args.model_save_path, X, batch, meta, len(samples), args).cpu().numpy()
    res = pd.DataFrame(pred)
    res.index = samples["patient"].to_list()
    res.to_csv(args.output)
    return res

def repeated_k_fold(df, meta, args):
    samples = df[["patient", "label"]].drop_duplicates()
    aucs = []
    for i in range(args.n_repeats):
        skf = StratifiedKFold(args.n_folds, shuffle=True, random_state=i)
        preds_ = []
        truths_ = []
        for train_idx, test_idx in skf.split(samples, samples["label"]):

            train_samples = samples.iloc[train_idx, :]
            test_samples = samples.iloc[test_idx, :]

            sampler = TPESampler(seed=0)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective_wrapper(df, meta, train_samples, args), n_trials=args.n_tune_trials)
            best_params = study.best_params

            X_test, y_test, batch_test, meta_test  = get_data(df, args.all_ct, test_samples, binary=args.binary, meta=meta if args.use_meta else None, attn2=args.attn2)

            X_train, y_train, batch_train, meta_train  = get_data(df, args.all_ct, train_samples, binary=args.binary, meta=meta if args.use_meta else None, attn2=args.attn2)

            model = train(X_train, y_train, batch_train, meta_train, args, dropout=best_params["dropout"], n_layers_lin=best_params["n_layers_lin"], n_layers_lin_meta=1 if not args.use_meta else best_params["n_layers_lin_meta"], n_hid=best_params["n_hid"], lr=best_params["lr"], weight_decay=best_params["weight_decay"], n_epochs=best_params["n_epochs"], seed=i)

            pred = predict(model, X_test, batch_test, meta_test, len(test_samples), args).cpu().numpy()

            truths_.extend(y_test)
            preds_.extend(pred)

        aucs.append(roc_auc_score(np.stack(truths_), np.stack(preds_), multi_class="ovo"))
    res = pd.DataFrame({"seed": range(args.n_repeats), "auc": aucs})
    res.to_csv(args.output)
    return np.mean(aucs), np.std(aucs)




