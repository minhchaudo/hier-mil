from utils import get_data, train, predict
from train import objective_wrapper
import itertools
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

import optuna as optuna
from optuna.samplers import TPESampler

def vary_train_size(train_sizes, df, args):
    samples = df[["patient", "label"]].drop_duplicates()
    mean_aucs_by_train_size = []
    for train_size in train_sizes:
        aucs = []
        for i in range(args.n_repeats):
            train_idx, test_idx = train_test_split(range(len(samples)), stratify=samples["label"].to_list(), train_size=train_size, random_state=i)
            train_samples = samples.iloc[train_idx, :]
            test_samples = samples.iloc[test_idx, :]

            sampler = TPESampler(seed=0)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective_wrapper(df, None, train_samples, args), n_trials=args.n_tune_trials)
            best_params = study.best_params

            X_test, y_test, batch_test, _  = get_data(df, args.all_ct, test_samples, binary=args.binary)

            X_train, y_train, batch_train, _  = get_data(df, args.all_ct, train_samples, binary=args.binary)

            model = train(X_train, y_train, batch_train, None, args, dropout=best_params["dropout"], n_layers_lin=best_params["n_layers_lin"], n_layers_lin_meta=1, n_hid=best_params["n_hid"], lr=best_params["lr"], weight_decay=best_params["weight_decay"], n_epochs=best_params["n_epochs"], seed=i)

            pred = predict(model, X_test, batch_test, None, len(test_samples), args).cpu().numpy()

            auc = roc_auc_score(y_test, pred, multi_class="ovo")
            aucs.append(auc)
        mean_aucs_by_train_size.append(np.mean(aucs))
    res = pd.DataFrame({"train_size": train_sizes, "mean_auc": mean_aucs_by_train_size})
    res.to_csv(args.output)
    return res


def vary_cell_count(cell_counts, df, args):
    samples = df[["patient", "label"]].drop_duplicates()
    mean_aucs_by_cell_count = []
    for cell_count in cell_counts:
        aucs = []
        for i in range(args.n_repeats):
            skf = StratifiedKFold(args.n_folds, shuffle=True, random_state=i)
            preds_ = []
            truths_ = []
            df_subsampled = df.groupby('patient', observed=False).apply(lambda x: x.sample(frac=cell_count, random_state=i), include_groups=True).reset_index(drop=True)
            for train_idx, test_idx in skf.split(samples, samples["label"]):

                train_samples = samples.iloc[train_idx, :]
                test_samples = samples.iloc[test_idx, :]

                sampler = TPESampler(seed=0)
                study = optuna.create_study(direction="maximize", sampler=sampler)
                study.optimize(objective_wrapper(df_subsampled, None, train_samples, args), n_trials=args.n_tune_trials)
                best_params = study.best_params

                X_train, y_train, batch_train, _  = get_data(df_subsampled, args.all_ct, train_samples, binary=args.binary)

                X_test, y_test, batch_test, _  = get_data(df_subsampled, args.all_ct, test_samples, binary=args.binary)

                model = train(X_train, y_train, batch_train, None, args, dropout=best_params["dropout"], n_layers_lin=best_params["n_layers_lin"], n_layers_lin_meta=1, n_hid=best_params["n_hid"], lr=best_params["lr"], weight_decay=best_params["weight_decay"], n_epochs=best_params["n_epochs"], seed=i)

                pred = predict(model, X_test, batch_test, None, len(test_samples), args).cpu().numpy()

                preds_.extend(pred)
                truths_.extend(y_test)
            aucs.append(roc_auc_score(np.stack(truths_), np.stack(preds_), multi_class="ovo"))
        mean_aucs_by_cell_count.append(np.mean(aucs))
    res = pd.DataFrame({"cell_count": cell_counts, "mean_auc": mean_aucs_by_cell_count})
    res.to_csv(args.output)
    return res

def reassign_cell_types(df, prop, all_ct, seed=0):
    np.random.seed(seed)

    df = df.copy()
    patients = df['patient'].unique()

    for patient in patients:
        patient_data = df[df['patient'] == patient]
        num_to_select = int(len(patient_data) * prop)

        selected_indices = np.random.choice(patient_data.index, num_to_select, replace=False)

        new_annotations = np.random.choice(all_ct, num_to_select, replace=True)

        df.loc[selected_indices, 'cell_type_annotation'] = new_annotations
    return df


def randomize_cell_annot(cell_props, df, args):
    samples = df[["patient", "label"]].drop_duplicates()
    mean_aucs_by_cell_prop = []
    for cell_prop in cell_props:
        aucs = []
        for i in range(args.n_repeats):
            skf = StratifiedKFold(args.n_folds, shuffle=True, random_state=i)
            preds_ = []
            truths_ = []
            for train_idx, test_idx in skf.split(samples, samples["label"]):

                df_subsampled = reassign_cell_types(df, cell_prop, args.all_ct, seed=i)
                train_samples = samples.iloc[train_idx, :]
                test_samples = samples.iloc[test_idx, :]

                sampler = TPESampler(seed=0)
                study = optuna.create_study(direction="maximize", sampler=sampler)
                study.optimize(objective_wrapper(df_subsampled, None, train_samples, args), n_trials=args.n_tune_trials)
                best_params = study.best_params


                X_train, y_train, batch_train, _  = get_data(df_subsampled, args.all_ct, train_samples, binary=args.binary)

                X_test, y_test, batch_test, _  = get_data(df_subsampled, args.all_ct, test_samples, binary=args.binary)

                model = train(X_train, y_train, batch_train, None, args, dropout=best_params["dropout"], n_layers_lin=best_params["n_layers_lin"], n_layers_lin_meta=1, n_hid=best_params["n_hid"], lr=best_params["lr"], weight_decay=best_params["weight_decay"], n_epochs=best_params["n_epochs"], seed=i)

                pred = predict(model, X_test, batch_test, None, len(test_samples), args).cpu().numpy()

                preds_.extend(pred)
                truths_.extend(y_test)

            aucs.append(roc_auc_score(np.stack(truths_), np.stack(preds_), multi_class="ovo"))
        mean_aucs_by_cell_prop.append(np.mean(aucs))

    res = pd.DataFrame({"cell_prop": cell_props, "mean_auc": mean_aucs_by_cell_prop})
    res.to_csv(args.output)
    return res




