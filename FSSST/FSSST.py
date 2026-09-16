from .preprocess import *
from .model import Encoder
from tqdm import tqdm
import torch.nn.functional as F


class FSSST():
    def __init__(self,
                 adata,
                 device=torch.device('cpu'),
                 learning_rate=0.001,
                 learning_rate_sc=0.01,
                 weight_decay=0.00,
                 epochs=600,
                 dim_output=64,
                 random_seed=41,
                 alpha=10,
                 gamma=1.0,
                 ):

        self.adata = adata.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.gamma = gamma

        fix_seed(self.random_seed)

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)

        construct_interaction(self.adata)

        if 'label_CSL' not in adata.obsm.keys():
            add_contrastive_label(self.adata)

        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.adj = self.adata.obsm['XY_adj']
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        coord = self.adata.obsm['spatial'].copy()
        self.coord = torch.FloatTensor(coord).to(self.device)


        self.adj = preprocess_adj(self.adj)
        self.adj = torch.FloatTensor(self.adj).to(self.device)

    def train(self):
        self.model = Encoder(self.dim_input, self.dim_output).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        print('Begin to train...')
        self.model.train()
        self.adata_copy = self.adata.copy()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            (laten_recon,
             node_feats_recon,
             output_recon, output_z, z, inr_Z, _,_) = self.model(self.features,
                                                            self.adj,
                                                            self.coord,
                                                            epoch=epoch,
                                                            start_update_epoch=1)
            self.loss_feat = F.mse_loss(self.features, laten_recon)
            self.loss_feat_1 = F.mse_loss(self.features, node_feats_recon)

            loss = (self.alpha * self.loss_feat +
                    self.gamma * self.loss_feat_1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            self.model.eval()
            laten_recon, xy_recon, output_d, output_z, z, inr_Z, corr_adj, corr_adj_p = self.model(self.features, self.adj,
                                                                             self.coord,
                                                                             epoch=self.epochs,
                                                                             start_update_epoch=1
                                                                             )
            self.emb_rec = output_d.detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec
            self.adata.obsm['corr_adj'] = corr_adj.detach().cpu().numpy()
            self.adata.obsm['corr_adj_p'] = corr_adj_p.detach().cpu().numpy()
        return self.adata

