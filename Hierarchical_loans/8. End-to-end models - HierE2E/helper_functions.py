
from NBEATS import NBEATS_PROFHIT
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from utilsforecast.losses import scaled_crps, smape, mase, mqloss, calibration
from functools import partial
from hierarchicalforecast.evaluation import evaluate



def quantiles_to_outputs(quantiles):
    output_names = []
    for q in quantiles:
        if q < 0.50:
            output_names.append(f"-lo-{np.round(100-200*q,2)}")
        elif q > 0.50:
            output_names.append(f"-hi-{np.round(100-200*(1-q),2)}")
        else:
            output_names.append("-median")
    return quantiles, output_names

def evaluation(method, models_dict, recon, seed, runs):
    df = pd.read_parquet('./loans_full_hierarchy.parquet')
    df['date'] = df.index
    df_nodate = df.reset_index(drop = True)
    levels = np.arange(5, 100, 5)
    tags = pd.read_pickle('./tags.pkl')
    Y_df = df.melt(id_vars='date')[['unique_id', 'date' ,'value']].rename(columns={'date' : 'ds', 'value' : 'y'})
    Y_df.ds = pd.to_datetime(Y_df.ds)
    Y_test_df = Y_df.groupby('unique_id', as_index=False).tail(24)
    Y_train_df = Y_df.drop(Y_test_df.index)

    i=0
    CRPS = []
    S = pd.read_pickle('./utils/full_BU_mat.pkl').cpu()
    map = pd.read_pickle('./utils/full_mapping.pkl')
    lev = [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]
    L = quantiles_to_outputs(levels/100)[1]
    for z,run in enumerate(runs) : 
        for q,model_name in enumerate(models_dict.keys()):
            for p,rec in enumerate(recon) :
                for k,splt in enumerate(seed):
                    # runid = mlflow.search_runs(experiment_ids=['2906'], filter_string=f'tags.mlflow.runName = "{method}_{model_name}_{rec}_12_{run}_{splt}"')['run_id'].values[0]
                    # test_CRPS = mlflow.search_runs(experiment_ids=['2906'], filter_string=f'tags.mlflow.runName = "{method}_{model_name}_{rec}_12_{run}_{splt}"')['metrics.test_CRPS'].values[0]
                    # if method == 'PROFHIT':
                    #     test_CRPS = mlflow.search_runs(experiment_ids=['2906'], filter_string=f'tags.mlflow.runName = "{method}_{model_name}_{rec}_12_{run}_{splt}"')['metrics.recon_loss'].values[0]
                    # CRPS.append(test_CRPS)
                    # path = f'./2906/{runid}/checkpoints/'
                    # final_path = path + os.listdir(path)[0]
                    final_path = f"./checkpoints/{method}_{model_name}_{rec}_12_{run}_{splt}"
                    model_prob = models_dict[model_name].load_from_checkpoint(final_path)
                    seq_len = 12   
                    Y_hat = pd.DataFrame()
                    cutoff = [df.index[-1] +  pd.Timedelta(days=1) - pd.DateOffset(months=(i)) - pd.Timedelta(days=1) for i in range(24,5, -1)]
                    for j,a in enumerate(cutoff):
                        cutin = df_nodate[df_nodate['date'] == a].index[0]
                        input = torch.tensor(df_nodate.iloc[cutin-(seq_len-1):cutin+1, :-1].values).unsqueeze(0)
                        true = torch.tensor(df_nodate.iloc[cutin+1:cutin+7, :-1].values).unsqueeze(0)
                        model_prob.eval()
                        with torch.no_grad():
                            if method == 'HierE2E' or method == 'PROFHIT':
                                rec_std, rec_mu, rec_samples = model_prob.forward_quantiles(batch =[input.float().to('cuda:0'),true.float().to('cuda:0')], batch_idx = 0, n_samples = 10000)
                            if method == 'CNF':
                                rec_samples = model_prob.forward_quantiles(batch =[input.float().to('cuda:0'),true.float().to('cuda:0')], batch_idx = 0, n_samples = 10000)
                            if method == 'CLOVER':
                                rec_mu, rec_std, _, rec_samples = model_prob.forward(batch =[input.float().to('cuda:0'),true.float().to('cuda:0')], batch_idx = 0, n_samples = 10000)

                        quantiles = torch.quantile(rec_samples.cpu().detach(),torch.tensor(levels/100).float(), dim = 0).squeeze()
                        # dist = torch.distributions.Normal(torch.zeros_like(rec_mu.cpu()),1.0)
                        columns = [f'{method}_{model_name}_{rec}_{splt}_{run}' + a for a in L ]
                        for i in range(126):
                            y_df = torch.permute(quantiles[:,:,i], (1,0))
                            y_df = pd.DataFrame(y_df)
                            y_df.columns = columns
                            unique_id = map
                            y_df['unique_id'] = unique_id[i]
                            y_df['ds'] = df.index[cutin+1:cutin+7]
                            y_df['cutoff'] = a
                            y_df['y'] = true[0,:,i]
                            y_df[f'{method}_{model_name}_{rec}_{splt}_{run}'] = y_df[f'{method}_{model_name}_{rec}_{splt}_{run}-median']
                            newcols = list(y_df.columns[-5:]) + list(y_df.columns[:-5])
                            y_df =y_df[newcols]
                            y_df = y_df.drop(columns = [f'{method}_{model_name}_{rec}_{splt}_{run}-median'])
                            if i == 0 and j == 0:
                                Y_hat = y_df
                            else:
                                Y_hat = pd.concat((Y_hat, y_df), axis = 0)
                    Y_hat = Y_hat.sort_values(by=['unique_id', 'cutoff'])
                    Y_hat.reset_index(drop = True, inplace = True)
                    if k == 0 and p == 0 and q == 0 and z == 0:
                        Y_fin = Y_hat
                    else :
                        Y_fin = pd.merge(Y_fin,Y_hat, on= ['unique_id', 'ds', 'cutoff', 'y'])
    import itertools
    model = [f'{method}_{model}_{rec}_{k}_{run}' for model,rec,k,run in itertools.product(models_dict.keys(), recon, seed, runs)]
    eval = evaluate(Y_fin,
            metrics = [scaled_crps, smape, partial(mase, seasonality = 6)], 
            tags = tags,
            train_df=Y_train_df,
            models = model,
                level=lev)
    return eval, Y_fin, CRPS