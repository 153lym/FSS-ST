import numpy as np
import pandas as pd
import torch
import os
import time
from FSSST.utils import clustering
import scanpy as sc
from FSSST import FSSST
from metrics import *


def do_model(datafile, alpha, gamma):
    adata = sc.read_h5ad(datafile)

    # define model
    model = FSSST.FSSST(adata, device=device, alpha=alpha, gamma=gamma)

    adata = model.train()
    return adata


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training model device on ..., ', device)

    datadir = '.../10X Visium-mouse liver/'  # dataset path
    filename_list = ['BH001']
    clusters_num_list = [4]
    alpha_list = [8.0]
    gamma_list = [6.0]
    tool = 'mclust'
    out_path = "results/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    local_time = time.localtime(time.time())
    run_all_re = []
    # 3. 格式化输出
    formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", local_time)
    for fname, cluster_num, alpha, gamma \
            in zip(filename_list,
                   clusters_num_list,
                   alpha_list,
                   gamma_list):
        print(fname, cluster_num, alpha, gamma, tool)
        adata = do_model(f'{datadir}/{fname}.h5ad',alpha, gamma)

        if tool == 'mclust':
            clustering(adata, cluster_num, radius=0, method=tool,
                       refinement=True)
        elif tool in ['leiden', 'louvain']:
            clustering(adata, cluster_num, radius=15, method=tool,
                       start=0.1, end=2.0,
                       increment=0.01, refinement=False)
        adata.write(f'{out_path}/{fname}.h5ad')
        adata = adata[np.logical_not(adata.obs['ground_truth'].isna())]  # remove NAN
        # compute ari
        ARI = compute_ARI(adata, 'ground_truth', f'domain')
        # compute nmi
        NMI = compute_NMI(adata, 'ground_truth', f'domain')
        print(fname, tool,alpha, gamma,ARI, NMI)
        run_all_re.append([fname, tool, alpha, gamma,ARI, NMI,])
        pd.DataFrame(run_all_re).to_csv(f'{out_path}/results.csv', index=False)
