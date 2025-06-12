import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import pandas as pd
import numpy as np
from einops import rearrange
from torch.distributions import Normal, Categorical
from neuralforecast.losses.pytorch import DistributionLoss, sCRPS
import pickle as pkl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import SymmetricMeanAbsolutePercentageError


def smape():
    return SymmetricMeanAbsolutePercentageError()

class HierE2E(L.LightningModule):
    def __init__(self,
                batch_size, test_batch_size,
                learning_rate,
                n_samples, criterion, quantiles, recon, dataset_type,
                **kwargs
                ):
        super().__init__()
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.dataset_type = dataset_type
        if self.dataset_type == 'geo':
            self.M = torch.tensor(pd.read_pickle(f'./utils/geo_proj_mat.pkl')).float().to('cuda:0')
            self.SG = pd.read_pickle(f'./utils/geo_BU_mat.pkl')
            self.SG = self.SG.float().to('cuda:0')
        elif self.dataset_type == 'full':
            self.M = torch.tensor(pd.read_pickle(f'./utils/full_proj_mat.pkl')).float().to('cuda:0')
            self.SG = pd.read_pickle(f'./utils/full_BU_mat.pkl')
            self.SG = self.SG.float().to('cuda:0')
        elif self.dataset_type == 'bad':
            self.SG = pd.read_pickle(f'./utils/bad_BU_mat.pkl')
            self.SG = self.SG.float().to('cuda:0')

        self.recon = recon
        self.n_samples = n_samples
        self.criterion = criterion
        self.NLloss = DistributionLoss('Normal')
        self.criterion_uni = smape()
        self.quantiles = quantiles
        self.CRPS = sCRPS(quantiles = quantiles)

        self.save_hyperparameters()
    def reconciliation(self, batch_std, batch_mu, n_samples, batch_size):
        
        dist = Normal(torch.zeros_like(batch_mu), 1.0)
        samples = dist.rsample((n_samples,))
        samples = samples * batch_std.repeat(n_samples,1,1,1)
        samples = samples + batch_mu.repeat(n_samples,1,1,1)
        if self.recon == 'proj':
            rec_samples = torch.einsum('iv ,sblv ->sbli', self.M, samples)
        elif self.recon == 'none':
            rec_samples = samples
        elif self.recon == 'BU':
            rec_samples = torch.einsum('iv ,sblv ->sbli', self.SG, samples)
        
        return rec_samples
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        batch_mu, batch_std = self.model(batch_x)
        #Reconciliation step
        rec_samples = self.reconciliation(batch_std, batch_mu, self.n_samples, self.batch_size)
        if self.criterion == 'likelihood':
            rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
            loss = self.NLloss(batch_y, [rec_mu, rec_std+1e-5])      
        elif self.criterion == 'sCRPS':
            rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
            dist = torch.distributions.Normal(torch.zeros_like(rec_mu),1.0)
            z = (batch_y-rec_mu)/rec_std
            sqrtpi = torch.sqrt(torch.tensor(torch.pi)).to('cuda:0')
            loss_CRPS = sqrtpi*torch.exp(dist.log_prob(z)) + (sqrtpi * z)/2*(2*dist.cdf(z)-1) + 1/2 * torch.log((2*rec_std)/sqrtpi)
            loss = torch.mean(loss_CRPS)   
        elif self.criterion == 'CRPS':
            quantiles = self.quantiles.float().to('cuda:0')
            quants = torch.quantile(rec_samples,quantiles, dim = 0)
            quants = torch.permute(quants, (1,2,3,0))
            error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
            CRPS = torch.sum((0.05)*torch.sum(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
            loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
            loss = torch.mean(loss_CRPS) 
        self.log(f'train_loss', loss, on_epoch = True) 
        return loss
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        batch_mu, batch_std = self.model(batch_x)
        #Reconciliation step
        rec_samples = self.reconciliation(batch_std, batch_mu, self.n_samples, self.test_batch_size)
        if self.criterion == 'likelihood':
            rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
            loss = self.NLloss(batch_y, [rec_mu, rec_std+1e-5])      
        elif self.criterion == 'sCRPS':
            rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
            dist = torch.distributions.Normal(torch.zeros_like(rec_mu),1.0)
            z = (batch_y-rec_mu)/rec_std
            sqrtpi = torch.sqrt(torch.tensor(torch.pi)).to('cuda:0')
            loss_CRPS = sqrtpi*torch.exp(dist.log_prob(z)) + (sqrtpi * z)/2*(2*dist.cdf(z)-1) + 1/2 * torch.log((2*rec_std)/sqrtpi)
            loss = torch.mean(loss_CRPS)   
        elif self.criterion == 'CRPS':
            quantiles = self.quantiles.float().to('cuda:0')
            quants = torch.quantile(rec_samples,quantiles, dim = 0)
            quants = torch.permute(quants, (1,2,3,0))
            error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
            CRPS = torch.sum((0.05)*torch.sum(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
            loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
            loss = torch.mean(loss_CRPS)   
        self.log(f'val_loss', loss) 

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_mu, batch_std = self.model(batch_x)
        #Reconciliation step
        
        rec_samples = self.reconciliation(batch_std, batch_mu, self.n_samples, self.test_batch_size)

        rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
        loss_uni = self.criterion_uni(batch_y,rec_mu)
        loss_likelihood = self.NLloss(batch_y, [rec_mu, rec_std])
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
        CRPS = torch.sum((0.05)*torch.sum(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
        loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
        loss_CRPS = torch.mean(loss_CRPS)
        self.log(f'test_CRPS', loss_CRPS) 
        dist = torch.distributions.Normal(torch.zeros_like(rec_mu),1.0)
        z = (batch_y-rec_mu)/rec_std
        sqrtpi = torch.sqrt(torch.tensor(torch.pi)).to('cuda:0')
        loss_CRPS = sqrtpi*torch.exp(dist.log_prob(z)) + (sqrtpi * z)/2*(2*dist.cdf(z)-1) + 1/2 * torch.log((2*rec_std)/sqrtpi)
        loss_CRPS = torch.mean(loss_CRPS)   
        self.log(f'test_sCRPS', loss_CRPS) 
        self.log(f'test_NLLoss', loss_likelihood)
        self.log('SMAPE_loss', loss_uni)
        if batch_idx == 0:
            self.x_test = batch_x.detach().cpu()
            self.y_test = batch_y.detach().cpu()
            self.mu_pred = rec_mu.detach().cpu()
            self.std_pred = rec_std.detach().cpu()
        else:
            self.x_test =  torch.cat((self.x_test,batch_x.detach().cpu()), dim=0)
            self.y_test =  torch.cat((self.y_test,batch_y.detach().cpu()), dim=0)
            self.mu_pred = torch.cat((self.mu_pred,rec_mu.detach().cpu()), dim=0)
            self.std_pred = torch.cat((self.std_pred,rec_std.detach().cpu()), dim=0)
    
    def on_test_epoch_end(self):
        '''
        Saves predictions in mlflow artifacts
        '''
        A = {'true' : np.array(self.y_test), 'point_pred' : np.array(self.mu_pred), 'std_pred' : np.array(self.std_pred), 'seq' : np.array(self.x_test)}
        pkl.dump(A, open(f'./{self.logger.experiment_id}/{self.logger.run_id}/preds_test_set.pkl', 'wb'))
        self.logger.experiment.log_artifact(self.logger.run_id, f'./{self.logger.experiment_id}/{self.logger.run_id}/preds_test_set.pkl')
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience = 10)
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                "scheduler": lr_scheduler,
                'monitor' : 'val_loss'
            }
        }


    def forward(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_mu, batch_std = self.model(batch_x)
        rec_samples = self.reconciliation(batch_std, batch_mu, self.n_samples, self.test_batch_size)
        rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        
        error = quants - batch_y.unsqueeze(-1).repeat(1,1,1,quantiles.shape[-1])
        sq = torch.maximum(-error, torch.zeros_like(error))[:,:,batch_idx,:]
        s1_q = torch.maximum(error, torch.zeros_like(error))[:,:,batch_idx,:]
        
        loss_CRPS = torch.mean((quantiles * sq+(1-quantiles) * s1_q))
        norm = torch.mean(torch.abs(batch_y[:,:,batch_idx]))
        loss_CRPS = 2*loss_CRPS / norm

        return rec_std, rec_mu, loss_CRPS
    
    def forward_quantiles(self, batch, batch_idx, n_samples):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_mu, batch_std = self.model(batch_x)
        if n_samples == None :
            n_samples = self.n_samples
        rec_samples = self.reconciliation(batch_std, batch_mu, n_samples, self.test_batch_size)
        rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        
        # error = quants - batch_y.unsqueeze(-1).repeat(1,1,1,quantiles.shape[-1])
        # sq = torch.maximum(-error, torch.zeros_like(error))[:,:,batch_idx,:]
        # s1_q = torch.maximum(error, torch.zeros_like(error))[:,:,batch_idx,:]
        
        # loss_CRPS = torch.mean((quantiles * sq+(1-quantiles) * s1_q))
        # norm = torch.mean(torch.abs(batch_y[:,:,batch_idx]))
        # loss_CRPS = 2*loss_CRPS / norm

        return rec_std, rec_mu, rec_samples
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('HierE2E')
        model_parser.add_argument('--learning_rate', type=float,default=0.0001)
        model_parser.add_argument('--criterion', type=str, choices = ['CRPS', 'likelihood'],default='likelihood')
        model_parser.add_argument('--quantiles', type=int,default=9)
        model_parser.add_argument('--n_samples', type=int,default=200)
        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        model_parser.add_argument('--recon', type=str, choices = ['none','BU', 'proj'], default='proj')


        return parent_parser
    
class PROFHIT(L.LightningModule):
    def __init__(self,
                batch_size, test_batch_size,
                learning_rate,
                n_samples, criterion, quantiles, lam, dataset_type,
                **kwargs
                ):
        super().__init__()
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.dataset_type = dataset_type
        if self.dataset_type == 'geo':
            self.SG = pd.read_pickle(f'./utils/geo_BU_mat.pkl')
            self.SG = self.SG.float().to('cuda:0')
        if self.dataset_type == 'full':
            self.SG = pd.read_pickle(f'./utils/full_BU_mat.pkl')
            self.SG = self.SG.float().to('cuda:0')
        self.lam = lam
        self.n_samples = n_samples
        self.criterion = criterion
        self.NLloss = DistributionLoss('Normal')
        self.criterion_uni = smape()
        self.quantiles = quantiles
        self.CRPS = sCRPS(quantiles = quantiles)
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        batch_mu, batch_std, rec_err = self.model(batch_x)
        if self.criterion == 'likelihood':
            loss = self.NLloss(batch_y, [batch_mu, batch_std])
            loss = loss + self.lam * torch.mean(rec_err)  
        elif self.criterion == 'CRPS':
            dist = Normal(torch.zeros_like(batch_mu), 1.0)
            samples = dist.rsample((self.n_samples,))
            samples = samples * batch_std.repeat(self.n_samples,1,1,1)
            samples = samples + batch_mu.repeat(self.n_samples,1,1,1)
            quantiles = self.quantiles.float().to('cuda:0')
            quants = torch.quantile(samples,quantiles, dim = 0)
            quants = torch.permute(quants, (1,2,3,0))
            # loss = self.CRPS(batch_y, quants, mask = torch.ones_like(batch_y))  
            z = (batch_y-batch_mu)/batch_std
            sqrtpi = torch.sqrt(torch.tensor(torch.pi)).to('cuda:0')
            loss_CRPS = sqrtpi*torch.exp(dist.log_prob(z)) + (sqrtpi * z)/2*(2*dist.cdf(z)-1) + 1/2 * torch.log((2*batch_std)/sqrtpi)
            loss = torch.mean(loss_CRPS)
            loss = loss + self.lam * torch.mean(rec_err)   
        self.log(f'train_loss', loss, on_epoch = True) 
        self.log(f'train_rec', torch.mean(rec_err), on_epoch = True) 
        return loss
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        batch_mu, batch_std, rec_err = self.model(batch_x)
        if self.criterion == 'likelihood':
            loss = self.NLloss(batch_y, [batch_mu, batch_std])  
            loss = loss + self.lam * torch.mean(rec_err)      
        elif self.criterion == 'CRPS':
            dist = Normal(torch.zeros_like(batch_mu), 1.0)
            samples = dist.rsample((self.n_samples,))
            samples = samples * batch_std.repeat(self.n_samples,1,1,1)
            samples = samples + batch_mu.repeat(self.n_samples,1,1,1)
            quantiles = self.quantiles.float().to('cuda:0')
            quants = torch.quantile(samples,quantiles, dim = 0)
            quants = torch.permute(quants, (1,2,3,0))
            z = (batch_y-batch_mu)/batch_std
            sqrtpi = torch.sqrt(torch.tensor(torch.pi)).to('cuda:0')
            loss_CRPS = sqrtpi*torch.exp(dist.log_prob(z)) + (sqrtpi * z)/2*(2*dist.cdf(z)-1) + 1/2 * torch.log((2*batch_std)/sqrtpi)
            loss = torch.mean(loss_CRPS)            
            loss = loss + self.lam * torch.mean(rec_err)   
        self.log(f'val_loss', loss) 
        self.log(f'val_rec', torch.mean(rec_err), on_epoch = True) 


    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_mu, batch_std, rec_err = self.model(batch_x)
        #Reconciliation step
        loss_uni = self.criterion_uni(batch_y,batch_mu)
        loss_likelihood = self.NLloss(batch_y, [batch_mu, batch_std])
        dist = Normal(torch.zeros_like(batch_mu), 1.0)
        samples = dist.rsample((self.n_samples,))
        samples = samples * batch_std.repeat(self.n_samples,1,1,1)
        samples = samples + batch_mu.repeat(self.n_samples,1,1,1)
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
        CRPS = torch.sum((0.05)*torch.sum(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
        loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
        loss_CRPS = torch.mean(loss_CRPS)
        self.log(f'test_CRPS', loss_CRPS) 

        z = (batch_y-batch_mu)/batch_std
        sqrtpi = torch.sqrt(torch.tensor(torch.pi)).to('cuda:0')
        loss_sCRPS = sqrtpi*torch.exp(dist.log_prob(z)) + (sqrtpi * z)/2*(2*dist.cdf(z)-1) + 1/2 * torch.log((2*batch_std)/sqrtpi)
        loss_sCRPS = torch.mean(loss_sCRPS)
        self.log(f'test_sCRPS', loss_sCRPS)
         
        self.log(f'test_NLLoss', loss_likelihood)
        self.log('SMAPE_loss', loss_uni)
        self.log('recon_loss', torch.mean(rec_err))
        if batch_idx == 0:
            self.x_test = batch_x.detach().cpu()
            self.y_test = batch_y.detach().cpu()
            self.mu_pred = batch_mu.detach().cpu()
            self.std_pred = batch_std.detach().cpu()
        else:
            self.x_test =  torch.cat((self.x_test,batch_x.detach().cpu()), dim=0)
            self.y_test =  torch.cat((self.y_test,batch_y.detach().cpu()), dim=0)
            self.mu_pred = torch.cat((self.mu_pred,batch_mu.detach().cpu()), dim=0)
            self.std_pred = torch.cat((self.std_pred,batch_std.detach().cpu()), dim=0)
    
    def on_test_epoch_end(self):
        '''
        Saves predictions in mlflow artifacts
        '''
        A = {'true' : np.array(self.y_test), 'point_pred' : np.array(self.mu_pred), 'std_pred' : np.array(self.std_pred), 'seq' : np.array(self.x_test)}
        pkl.dump(A, open(f'./{self.logger.experiment_id}/{self.logger.run_id}/preds_test_set.pkl', 'wb'))
        self.logger.experiment.log_artifact(self.logger.run_id, f'./{self.logger.experiment_id}/{self.logger.run_id}/preds_test_set.pkl')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience = 10)
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                "scheduler": lr_scheduler,
                'monitor' : 'val_loss'
                }
        }


    def forward(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_mu, batch_std = self.model(batch_x)
        rec_samples = self.reconciliation(batch_std, batch_mu, self.n_samples, self.test_batch_size)
        rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        
        error = quants - batch_y.unsqueeze(-1).repeat(1,1,1,quantiles.shape[-1])
        sq = torch.maximum(-error, torch.zeros_like(error))[:,:,batch_idx,:]
        s1_q = torch.maximum(error, torch.zeros_like(error))[:,:,batch_idx,:]
        
        loss_CRPS = torch.mean((quantiles * sq+(1-quantiles) * s1_q))
        norm = torch.mean(torch.abs(batch_y[:,:,batch_idx]))
        loss_CRPS = 2*loss_CRPS / norm

        return rec_std, rec_mu, loss_CRPS
    
    def forward_quantiles(self, batch, batch_idx, n_samples):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_mu, batch_std, rec_err = self.model(batch_x)
        if n_samples == None :
            n_samples = self.n_samples
        dist = Normal(torch.zeros_like(batch_mu), 1.0)
        samples = dist.rsample((n_samples,))
        samples = samples * batch_std.repeat(n_samples,1,1,1)
        samples = samples + batch_mu.repeat(n_samples,1,1,1)
        return batch_std, batch_mu, samples
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('HierE2E')
        model_parser.add_argument('--learning_rate', type=float,default=0.0001)
        model_parser.add_argument('--criterion', type=str, choices = ['CRPS', 'likelihood'],default='likelihood')
        model_parser.add_argument('--quantiles', type=int,default=9)
        model_parser.add_argument('--n_samples', type=int,default=200)
        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        model_parser.add_argument('--lam', type=float, default=0.5)


        return parent_parser

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, mode = 'revin'):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine == 1:
            self._init_params()
        self.type = mode

    def forward(self, x, mode:str):
            if mode == 'norm':
                self._get_statistics(x)
                x = self._normalize(x)
            elif mode == 'denorm':
                x = self._denormalize(x)
            elif mode == 'denorm_scale':
                x = self._denormalize_scale(x)
            return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        if self.type == 'revin':
            dim2reduce = tuple(range(1, x.ndim-1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            if self.affine :    
                self.mean = self.mean + self.affine_bias
                self.stdev = self.stdev * (torch.relu(self.affine_weight) + self.eps)
        elif self.type == 'robust':
            dim2reduce = tuple(range(1, x.ndim-1))
            self.mean = torch.median(x, dim=1, keepdim=True).values.detach()
            x_mad = torch.median(torch.abs(x-self.mean), dim=1, keepdim = True).values.detach()
            stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            x_mad_aux = stdev * 0.6744897501960817
            x_mad = x_mad * (x_mad>0) + x_mad_aux * (x_mad==0)
            x_mad[x_mad==0] = 1.0
            x_mad = x_mad + self.eps
            self.stdev = x_mad
            if self.affine :    
                self.mean = self.mean + self.affine_bias
                self.stdev = self.stdev * (torch.relu(self.affine_weight) + self.eps)
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x
    
    def _denormalize_scale(self, x, eps = 1e-5):  
        x = x * self.stdev
        return x
    def robust_statistics(self, x, dim=-1, eps=1e-6):
        return None

class Refinement(nn.Module):
    '''
    Refinement module from PROFHIT paper
    '''
    def __init__(self, enc_in: int, c: float = 5.0) -> None:
        super(Refinement, self).__init__()
        self.enc_in = enc_in
        self.c = c
        self.w_hat = nn.Parameter(
            torch.randn(self.enc_in)/ np.sqrt(self.enc_in) + c
        )
        self.w = nn.Linear(self.enc_in, self.enc_in)
        self.b = nn.Parameter(
            torch.randn(self.enc_in) / np.sqrt(self.enc_in) + c
        )
        self.v1 = nn.Linear(self.enc_in, self.enc_in)
        self.v2 = nn.Linear(self.enc_in, self.enc_in)

    def forward(self, mu, logstd):
        gamma = torch.sigmoid(self.w_hat)
        mu_final = gamma * mu + (1 - gamma) * self.w(mu)
        logstd_final = torch.sigmoid(self.b) * logstd + (
            1.0 - torch.sigmoid(self.b)
        ) * (self.v1(mu) + self.v2(logstd))
        return mu_final, logstd_final
    

class CLOVER(L.LightningModule):
    def __init__(self,
                batch_size, test_batch_size,
                learning_rate,
                n_samples, quantiles, n_components, criterion, dataset_type,
                **kwargs
                ):
        super().__init__()
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.criterion = criterion
        self.n_components = n_components
        self.test_batch_size = test_batch_size
        self.dataset_type = dataset_type
        if self.dataset_type == 'geo':
            self.n_high = 6
            self.SG = pd.read_pickle(f'./utils/geo_BU_mat.pkl')[:,self.n_high:]
            self.SG = self.SG.float().to('cuda:0')
        elif self.dataset_type == 'full':
            self.n_high = 26
            self.SG = pd.read_pickle(f'./utils/full_BU_mat.pkl')[:,self.n_high:]
            self.SG = self.SG.float().to('cuda:0')
        elif self.dataset_type == 'bad':
            self.n_high = 26
            self.SG = pd.read_pickle(f'./utils/bad_BU_mat.pkl')[:,self.n_high:]
            self.SG = self.SG.float().to('cuda:0')

        self.n_samples = n_samples
        self.NLloss = DistributionLoss('Normal')
        self.criterion_uni = smape()
        self.quantiles = quantiles
        self.CRPS = sCRPS(quantiles = quantiles)

        self.save_hyperparameters()

    def probBU(self, batch_std, batch_mu, batch_factor, n_samples):
        batch_factor = torch.einsum('iv ,blvf ->blif', self.SG, batch_factor)
        batch_mu = torch.einsum('iv ,blv ->bli', self.SG, batch_mu)
        batch_std = torch.einsum('iv ,blv ->bli', self.SG, batch_std)
        factor_dist = Normal(loc = torch.zeros((batch_factor.shape[0], batch_factor.shape[1], batch_factor.shape[-1])).to('cuda:0'), scale = 1.0)
        norm_dist = Normal(loc = torch.zeros_like(batch_mu), scale = 1.0)
        factor_samples = factor_dist.rsample((n_samples, ))
        factor_samples = torch.einsum('bpmc ,sbpc -> sbpm', batch_factor, factor_samples) 
        samples = norm_dist.rsample((n_samples, ))
        samples = F.relu(batch_mu.repeat(n_samples,1,1,1) +
                         samples * batch_std.repeat(n_samples,1,1,1) + 
                         factor_samples)
        return samples

    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x[:,:,self.n_high:].float()
        batch_y = batch_y.float()

        batch_mu, batch_std, batch_factor = self.model(batch_x)
        #Reconciliation step
        if self.criterion == 'CRPS':
            rec_samples = self.probBU(batch_std, batch_mu, batch_factor, self.n_samples)
            quantiles = self.quantiles.float().to('cuda:0')
            quants = torch.quantile(rec_samples,quantiles, dim = 0)
            quants = torch.permute(quants, (1,2,3,0))
            error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
            CRPS = torch.sum(torch.mean(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
            loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
            # loss_CRPS = CRPS
            loss = torch.mean(loss_CRPS)
        if self.criterion == 'sCRPS':
            rec_samples = self.probBU(batch_std, batch_mu, batch_factor, self.n_samples)
            loss = (rec_samples - batch_y).abs().mean(0) 
            incons = (rec_samples - rec_samples.unsqueeze(1)).abs().mean([0, 1])
            sCRPS = loss/incons + (1/2)*torch.log(incons)
            loss = torch.mean(sCRPS)
        elif self.criterion == 'likelihood':
            batch_f = torch.einsum('iv ,blvf ->blif', self.SG, batch_factor)
            batch_mu = torch.einsum('iv ,blv ->bli', self.SG, batch_mu)
            batch_std = torch.einsum('iv ,blv ->bli', self.SG, batch_std)
            sig = torch.diag_embed(torch.square(batch_std), dim1 = -2, dim2 = -1)
            F = batch_f @ batch_f.permute((0,1,3,2))
            mult_norm = torch.distributions.MultivariateNormal(batch_mu, sig+F)
            loss = - torch.mean(mult_norm.log_prob(batch_y))
        self.log(f'train_loss', loss, on_epoch = True) 
        return loss
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x[:,:,self.n_high:].float()
        batch_y = batch_y.float()

        batch_mu, batch_std, batch_factor = self.model(batch_x)
        #Reconciliation step
        if self.criterion == 'CRPS':
            rec_samples = self.probBU(batch_std, batch_mu, batch_factor, self.n_samples)
            quantiles = self.quantiles.float().to('cuda:0')
            quants = torch.quantile(rec_samples,quantiles, dim = 0)
            quants = torch.permute(quants, (1,2,3,0))
            error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
            CRPS = torch.sum(torch.mean(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
            loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
            # loss_CRPS = CRPS
            loss = torch.mean(loss_CRPS)
        if self.criterion == 'sCRPS':
            rec_samples = self.probBU(batch_std, batch_mu, batch_factor, self.n_samples)
            loss = (rec_samples - batch_y).abs().mean(0) 
            incons = (rec_samples - rec_samples.unsqueeze(1)).abs().mean([0, 1])
            sCRPS = loss/incons + (1/2)*torch.log(incons)
            loss = torch.mean(sCRPS)
        elif self.criterion == 'likelihood':
            batch_f = torch.einsum('iv ,blvf ->blif', self.SG, batch_factor)
            batch_mu = torch.einsum('iv ,blv ->bli', self.SG, batch_mu)
            batch_std = torch.einsum('iv ,blv ->bli', self.SG, batch_std)
            sig = torch.diag_embed(torch.square(batch_std), dim1 = -2, dim2 = -1)
            F = batch_f @ batch_f.permute((0,1,3,2))
            mult_norm = torch.distributions.MultivariateNormal(batch_mu, sig+F)
            loss = - torch.mean(mult_norm.log_prob(batch_y))
        self.log(f'val_loss', loss) 

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x[:,:,self.n_high:].float()
        batch_y = batch_y.float()
        batch_mu, batch_std, batch_factor = self.model(batch_x)
        #Reconciliation step
        rec_samples = self.probBU(batch_std, batch_mu, batch_factor, self.n_samples)
        rec_mu = torch.mean(rec_samples, dim = 0)
        loss_uni = self.criterion_uni(batch_y,rec_mu)
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
        CRPS = torch.sum(torch.mean(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
        loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
        loss_CRPS = torch.mean(loss_CRPS)
        loss = (rec_samples - batch_y).abs().mean(0) 
        incons = (rec_samples - rec_samples.unsqueeze(1)).abs().mean([0, 1])
        sCRPS = loss/incons + (1/2)*torch.log(incons)
        batch_f = torch.einsum('iv ,blvf ->blif', self.SG, batch_factor)
        batch_mu = torch.einsum('iv ,blv ->bli', self.SG, batch_mu)
        batch_std = torch.einsum('iv ,blv ->bli', self.SG, batch_std)
        sig = torch.diag_embed(torch.square(batch_std), dim1 = -2, dim2 = -1)
        F = batch_f @ batch_f.permute((0,1,3,2))
        mult_norm = torch.distributions.MultivariateNormal(batch_mu, sig+F)
        lossNLL = - torch.mean(mult_norm.log_prob(batch_y))
        loss_sCRPS = torch.mean(sCRPS)
        self.log(f'test_NLLoss', lossNLL) 
        self.log(f'test_sCRPS', loss_sCRPS) 
        self.log(f'test_CRPS', loss_CRPS) 
        self.log('SMAPE_loss', loss_uni)
        if batch_idx == 0:
            self.x_test = batch_x.detach().cpu()
            self.y_test = batch_y.detach().cpu()
            self.mu_pred = rec_mu.detach().cpu()
        else:
            self.x_test =  torch.cat((self.x_test,batch_x.detach().cpu()), dim=0)
            self.y_test =  torch.cat((self.y_test,batch_y.detach().cpu()), dim=0)
            self.mu_pred = torch.cat((self.mu_pred,rec_mu.detach().cpu()), dim=0)
    
    def on_test_epoch_end(self):
        '''
        Saves predictions in mlflow artifacts
        '''
        A = {'true' : np.array(self.y_test), 'point_pred' : np.array(self.mu_pred), 'seq' : np.array(self.x_test)}
        pkl.dump(A, open(f'./{self.logger.experiment_id}/{self.logger.run_id}/preds_test_set.pkl', 'wb'))
        self.logger.experiment.log_artifact(self.logger.run_id, f'./{self.logger.experiment_id}/{self.logger.run_id}/preds_test_set.pkl')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience = 10)
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                "scheduler": lr_scheduler,
                'monitor' : 'val_loss'
                }
        }

    
    def forward(self, batch, batch_idx, n_samples):
        batch_x, batch_y = batch
        batch_x = batch_x[:,:,self.n_high:].float()
        batch_y = batch_y.float()
        batch_mu, batch_std, batch_factor = self.model(batch_x)
        #Reconciliation step
        if n_samples == None :
            n_samples = self.n_samples
        rec_samples = self.probBU(batch_std, batch_mu, batch_factor, n_samples)
        rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
        return batch_mu, batch_std, batch_factor,  rec_samples
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('CLOVER')
        model_parser.add_argument('--learning_rate', type=float,default=0.0001)
        model_parser.add_argument('--quantiles', type=int,default=9)
        model_parser.add_argument('--n_samples', type=int,default=200)
        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        model_parser.add_argument('--n_components', type=int, default = 5)
        model_parser.add_argument('--criterion', type=str, default = 'CRPS')
        return parent_parser
    
class DPMN(L.LightningModule):
    def __init__(self,
                batch_size, test_batch_size,
                learning_rate,
                n_samples, criterion, quantiles, recon, dataset_type, n_components,
                **kwargs
                ):
        super().__init__()
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.dataset_type = dataset_type
        self.n_components = n_components
        if self.dataset_type == 'geo':
            self.M = torch.tensor(pd.read_pickle(f'./utils/geo_proj_mat.pkl')).float().to('cuda:0')
            self.SG = pd.read_pickle(f'./utils/geo_BU_mat.pkl')
            self.SG = self.SG.float().to('cuda:0')
        elif self.dataset_type == 'full':
            # self.M = torch.tensor(pd.read_pickle(f'./utils/geo_proj_mat.pkl')).float().to('cuda:0')
            self.SG = pd.read_pickle(f'./utils/full_BU_mat.pkl')
            self.SG = self.SG.float().to('cuda:0')
        elif self.dataset_type == 'bad':
            # self.M = torch.tensor(pd.read_pickle(f'./utils/geo_proj_mat.pkl')).float().to('cuda:0')
            self.SG = pd.read_pickle(f'./utils/bad_BU_mat.pkl')
            self.SG = self.SG.float().to('cuda:0')

        self.recon = recon
        self.n_samples = n_samples
        self.criterion = criterion
        self.NLloss = DistributionLoss('Normal')
        self.criterion_uni = smape()
        self.quantiles = quantiles
        self.CRPS = sCRPS(quantiles = quantiles)

        self.save_hyperparameters()
        
    def reconciliation(self, batch_std, batch_mu, n_samples, batch_size):
        
        dist = Normal(torch.zeros_like(batch_mu[:,:,:,0]), 1.0)
        cat = Categorical(torch.full_like(batch_mu, fill_value= 1/self.n_components))
        cat_samples = cat.sample((n_samples,))
        cat_samples = torch.eye(self.n_components).to('cuda:0')[cat_samples]
        batch_std = torch.einsum('csvbp, csvbp -> csvb', cat_samples, batch_std.repeat(n_samples,1,1,1,1))
        batch_mu = torch.einsum('csvbp, csvbp -> csvb', cat_samples, batch_mu.repeat(n_samples,1,1,1,1))
        samples = dist.rsample((n_samples,))
        samples = samples * batch_std
        samples = samples + batch_mu
        if self.recon == 'proj':
            rec_samples = torch.einsum('iv ,sblv ->sbli', self.M, samples)
        elif self.recon == 'none':
            rec_samples = samples
        elif self.recon == 'BU':
            rec_samples = torch.einsum('iv ,sblv ->sbli', self.SG, samples)
        
        return rec_samples
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        batch_mu, batch_std = self.model(batch_x)
        #Reconciliation step
        rec_samples = self.reconciliation(batch_std, batch_mu, self.n_samples, self.batch_size)
        if self.criterion == 'likelihood':
            rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
            loss = self.NLloss(batch_y, [rec_mu, rec_std+1e-5])      
        elif self.criterion == 'CRPS':
            quantiles = self.quantiles.float().to('cuda:0')
            quants = torch.quantile(rec_samples,quantiles, dim = 0)
            quants = torch.permute(quants, (1,2,3,0))
            error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
            CRPS = torch.sum(torch.mean(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
            loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
            loss = torch.mean(loss_CRPS)
        self.log(f'train_loss', loss, on_epoch = True) 
        return loss
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        batch_mu, batch_std = self.model(batch_x)
        #Reconciliation step
        rec_samples = self.reconciliation(batch_std, batch_mu, self.n_samples, self.test_batch_size)
        if self.criterion == 'likelihood':
            rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
            loss = self.NLloss(batch_y, [rec_mu, rec_std+1e-5])      
        elif self.criterion == 'CRPS':
            quantiles = self.quantiles.float().to('cuda:0')
            quants = torch.quantile(rec_samples,quantiles, dim = 0)
            quants = torch.permute(quants, (1,2,3,0))
            error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
            CRPS = torch.sum(torch.mean(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
            loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
            loss = torch.mean(loss_CRPS)
        self.log(f'val_loss', loss) 

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_mu, batch_std = self.model(batch_x)
        #Reconciliation step
        
        rec_samples = self.reconciliation(batch_std, batch_mu, self.n_samples, self.test_batch_size)

        rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
        loss_uni = self.criterion_uni(batch_y,rec_mu)
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
        CRPS = torch.sum(torch.mean(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
        loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
        self.log(f'test_CRPS', torch.mean(loss_CRPS))
        # dist = torch.distributions.Normal(torch.zeros_like(rec_mu),1.0)
        # z = (batch_y-rec_mu)/rec_std
        # sqrtpi = torch.sqrt(torch.tensor(torch.pi)).to('cuda:0')
        # loss_CRPS = sqrtpi*torch.exp(dist.log_prob(z)) + (sqrtpi * z)/2*(2*dist.cdf(z)-1) + 1/2 * torch.log((2*rec_std)/sqrtpi)
        # loss_CRPS = torch.mean(loss_CRPS)   
        # self.log(f'test_sCRPS', loss_CRPS) 
        self.log('SMAPE_loss', loss_uni)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience = 10)
        return {
            'optimizer' : optimizer,
            # 'lr_scheduler' : {
            #     "scheduler": lr_scheduler,
            #     'monitor' : 'val_loss'
            # }
        }


    def forward(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_mu, batch_std = self.model(batch_x)
        rec_samples = self.reconciliation(batch_std, batch_mu, self.n_samples, self.test_batch_size)
        rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        
        error = quants - batch_y.unsqueeze(-1).repeat(1,1,1,quantiles.shape[-1])
        sq = torch.maximum(-error, torch.zeros_like(error))[:,:,batch_idx,:]
        s1_q = torch.maximum(error, torch.zeros_like(error))[:,:,batch_idx,:]
        
        loss_CRPS = torch.mean((quantiles * sq+(1-quantiles) * s1_q))
        norm = torch.mean(torch.abs(batch_y[:,:,batch_idx]))
        loss_CRPS = 2*loss_CRPS / norm

        return rec_std, rec_mu, loss_CRPS
    
    def forward_quantiles(self, batch, batch_idx, n_samples):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_mu, batch_std = self.model(batch_x)
        if n_samples == None :
            n_samples = self.n_samples
        rec_samples = self.reconciliation(batch_std, batch_mu, n_samples, self.test_batch_size)
        rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        
        # error = quants - batch_y.unsqueeze(-1).repeat(1,1,1,quantiles.shape[-1])
        # sq = torch.maximum(-error, torch.zeros_like(error))[:,:,batch_idx,:]
        # s1_q = torch.maximum(error, torch.zeros_like(error))[:,:,batch_idx,:]
        
        # loss_CRPS = torch.mean((quantiles * sq+(1-quantiles) * s1_q))
        # norm = torch.mean(torch.abs(batch_y[:,:,batch_idx]))
        # loss_CRPS = 2*loss_CRPS / norm

        return rec_std, rec_mu, rec_samples
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('DPMN')
        model_parser.add_argument('--learning_rate', type=float,default=0.0001)
        model_parser.add_argument('--criterion', type=str, choices = ['sCRPS', 'CRPS', 'likelihood'],default='likelihood')
        model_parser.add_argument('--quantiles', type=int,default=9)
        model_parser.add_argument('--n_samples', type=int,default=200)
        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        model_parser.add_argument('--recon', type=str, choices = ['none','BU', 'proj'], default='proj')
        model_parser.add_argument('--n_components', type=int, default = 5)
        return parent_parser
    


class CNF(L.LightningModule):
    def __init__(self,
                batch_size, test_batch_size,
                learning_rate,
                n_samples, quantiles, dataset_type, num_blocks,
                **kwargs
                ):
        super().__init__()
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_blocks = num_blocks
        self.test_batch_size = test_batch_size
        self.dataset_type = dataset_type
        if self.dataset_type == 'geo':
            self.n_high = 6
            self.SG = pd.read_pickle(f'./utils/geo_BU_mat.pkl')[:,self.n_high:]
            self.SG = self.SG.float().to('cuda:0')
        elif self.dataset_type == 'full':
            self.n_high = 26
            self.SG = pd.read_pickle(f'./utils/full_BU_mat.pkl')[:,self.n_high:]
            self.SG = self.SG.float().to('cuda:0')
        elif self.dataset_type == 'bad':
            self.n_high = 26
            self.SG = pd.read_pickle(f'./utils/bad_BU_mat.pkl')[:,self.n_high:]
            self.SG = self.SG.float().to('cuda:0')

        self.n_samples = n_samples
        self.NLloss = DistributionLoss('Normal')
        self.criterion_uni = smape()
        self.quantiles = quantiles
        self.CRPS = sCRPS(quantiles = quantiles)

        self.save_hyperparameters()
        
    def reconciliation(self,samples):
    
        rec_samples = torch.einsum('iv ,sblv ->sbli', self.SG, samples)
        
        return rec_samples
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        forecast_samples = self.model(batch_x)
        #Reconciliation step
        rec_samples = self.reconciliation(forecast_samples)
    
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
        CRPS = torch.sum((0.05)*torch.sum(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
        loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
        loss = torch.mean(loss_CRPS) 
        self.log(f'train_loss', loss, on_epoch = True) 
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        forecast_samples = self.model(batch_x)
        #Reconciliation step
        rec_samples = self.reconciliation(forecast_samples)
    
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
        CRPS = torch.sum((0.05)*torch.sum(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
        loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
        loss = torch.mean(loss_CRPS) 
        self.log(f'val_loss', loss) 

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        
        forecast_samples = self.model(batch_x)
        #Reconciliation step
        rec_samples = self.reconciliation(forecast_samples)
        
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        error = batch_y.unsqueeze(-1).repeat(1,1,1,quants.shape[-1]) - quants
        CRPS = torch.sum((0.05)*torch.sum(torch.maximum(quantiles * error,(quantiles- 1) * error), dim = -1), dim = 1)
        loss_CRPS = 2 * CRPS/torch.sum(torch.abs(batch_y), dim = 1)
        loss_CRPS = torch.mean(loss_CRPS)
        self.log(f'test_CRPS', loss_CRPS) 
        _, rec_mu = torch.std_mean(rec_samples, dim = 0)
        loss_uni = self.criterion_uni(batch_y,rec_mu)
        self.log('SMAPE_loss', loss_uni)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience = 10)
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                "scheduler": lr_scheduler,
                'monitor' : 'val_loss'
            }
        }


    def forward(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_mu, batch_std = self.model(batch_x)
        rec_samples = self.reconciliation(batch_std, batch_mu, self.n_samples, self.test_batch_size)
        rec_std, rec_mu = torch.std_mean(rec_samples, dim = 0)
        quantiles = self.quantiles.float().to('cuda:0')
        quants = torch.quantile(rec_samples,quantiles, dim = 0)
        quants = torch.permute(quants, (1,2,3,0))
        
        error = quants - batch_y.unsqueeze(-1).repeat(1,1,1,quantiles.shape[-1])
        sq = torch.maximum(-error, torch.zeros_like(error))[:,:,batch_idx,:]
        s1_q = torch.maximum(error, torch.zeros_like(error))[:,:,batch_idx,:]
        
        loss_CRPS = torch.mean((quantiles * sq+(1-quantiles) * s1_q))
        norm = torch.mean(torch.abs(batch_y[:,:,batch_idx]))
        loss_CRPS = 2*loss_CRPS / norm

        return rec_std, rec_mu, loss_CRPS
    
    def forward_quantiles(self, batch, batch_idx, n_samples):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
      
        forecast_samples = self.model(batch_x)
        #Reconciliation step
        rec_samples = self.reconciliation(forecast_samples)
        return rec_samples
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('HierE2E')
        model_parser.add_argument('--learning_rate', type=float,default=0.0001)
        model_parser.add_argument('--quantiles', type=int,default=9)
        model_parser.add_argument('--n_samples', type=int,default=200)
        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        model_parser.add_argument('--num_blocks', type=int, default=5)


        return parent_parser
