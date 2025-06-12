
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import HierE2E, RevIN, PROFHIT, CLOVER, Refinement



class _IdentityBasis(nn.Module):
    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        interpolation_mode: str,
        out_features: int = 1,
    ):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or (
            "cubic" in interpolation_mode
        )
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode
        self.out_features = out_features

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :]

        # Interpolation is performed on default dim=-1 := H
        knots = knots.reshape(len(knots), self.out_features, -1)
        if self.interpolation_mode in ["nearest", "linear"]:
            # knots = knots[:,None,:]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )
            # forecast = forecast[:,0,:]
        elif "cubic" in self.interpolation_mode:
            if self.out_features > 1:
                raise Exception(
                    "Cubic interpolation not available with multiple outputs."
                )
            batch_size = len(backcast)
            knots = knots[:, None, :, :]
            forecast = torch.zeros(
                (len(knots), self.forecast_size), device=knots.device
            )
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size : (i + 1) * batch_size],
                    size=self.forecast_size,
                    mode="bicubic",
                )
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[
                    :, 0, 0, :
                ]  # [B,None,H,H] -> [B,H]
            forecast = forecast[:, None, :]  # [B,H] -> [B,None,H]

        # [B,Q,H] -> [B,H,Q]
        forecast = forecast.permute(0, 2, 1)
        return backcast, forecast


ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]

POOLING = ["MaxPool1d", "AvgPool1d"]


class NHITSBlock(nn.Module):
    """
    NHITS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        futr_input_size: int,
        hist_input_size: int,
        stat_input_size: int,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout_prob: float,
        activation: str,
    ):
        super().__init__()

        pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))
        pooled_futr_size = int(np.ceil((input_size + h) / n_pool_kernel_size))

        input_size = (
            pooled_hist_size
            + hist_input_size * pooled_hist_size
            + futr_input_size * pooled_futr_size
            + stat_input_size
        )

        self.dropout_prob = dropout_prob
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        assert pooling_mode in POOLING, f"{pooling_mode} is not in {POOLING}"

        activ = getattr(nn, activation)()

        self.pooling_layer = getattr(nn, pooling_mode)(
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )

        # Block MLPs
        hidden_layers = [
            nn.Linear(in_features=input_size, out_features=mlp_units[0][0])
        ]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hidden_layers.append(activ)

            if self.dropout_prob > 0:
                # raise NotImplementedError('dropout')
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(
        self,
        insample_y: torch.Tensor,
        futr_exog: torch.Tensor,
        hist_exog: torch.Tensor,
        stat_exog: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Pooling
        # Pool1d needs 3D input, (B,C,L), adding C dimension
        insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        # Flatten MLP inputs [B, L+H, C] -> [B, (L+H)*C]
        # Contatenate [ Y_t, | X_{t-L},..., X_{t} | F_{t-L},..., F_{t+H} | S ]
        batch_size = len(insample_y)
        if self.hist_input_size > 0:
            hist_exog = hist_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            hist_exog = self.pooling_layer(hist_exog)
            hist_exog = hist_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            insample_y = torch.cat(
                (insample_y, hist_exog.reshape(batch_size, -1)), dim=1
            )

        if self.futr_input_size > 0:
            futr_exog = futr_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            futr_exog = self.pooling_layer(futr_exog)
            futr_exog = futr_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            insample_y = torch.cat(
                (insample_y, futr_exog.reshape(batch_size, -1)), dim=1
            )

        if self.stat_input_size > 0:
            insample_y = torch.cat(
                (insample_y, stat_exog.reshape(batch_size, -1)), dim=1
            )

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


class NHITS(nn.Module):
    def __init__(
        self,
        h,
        input_size,
        stack_types: list = ["identity", "identity", "identity"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        n_pool_kernel_size: list = [2, 2, 1],
        n_freq_downsample: list = [4, 2, 1],
        pooling_mode: str = "MaxPool1d",
        interpolation_mode: str = "linear",
        dropout_prob_theta: float =0.0,
        activation: str = "ReLU",
        affine: bool = True,
        scaler: str = 'revin',
        enc_in: int = 26,
        method: str = 'HierE2E',
        S_mat = None,
        n_components: int = 5,
    ):

        # Inherit BaseWindows class
        super().__init__()
        self.enc_in = enc_in
        self.revin = RevIN(self.enc_in, affine = affine, mode=scaler)
        self.std_activ = nn.Softplus()
        self.outputsize_multiplier = 2
        self.h = h
        self.n_components = n_components
        if method == 'PROFHIT':
            self.refine = Refinement(enc_in=self.enc_in)
        if method == 'CLOVER':
            self.outputsize_multiplier = 2 + self.n_components
        self.S_mat = S_mat
        self.method = method
        self.n_components = n_components
        self.decompose_forecast = False
        self.input_size = input_size
        self.futr_exog_size = 0
        self.hist_exog_size = 0
        self.stat_exog_size = 0
        # Architecture
        blocks = self.create_stack(
            h=h,
            input_size=input_size,
            stack_types=stack_types,
            futr_input_size=self.futr_exog_size,
            hist_input_size=self.hist_exog_size,
            stat_input_size=self.stat_exog_size,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            n_pool_kernel_size=n_pool_kernel_size,
            n_freq_downsample=n_freq_downsample,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout_prob_theta=dropout_prob_theta,
            activation=activation,
        )
        self.blocks = torch.nn.ModuleList(blocks)

    def create_stack(
        self,
        h,
        input_size,
        stack_types,
        n_blocks,
        mlp_units,
        n_pool_kernel_size,
        n_freq_downsample,
        pooling_mode,
        interpolation_mode,
        dropout_prob_theta,
        activation,
        futr_input_size,
        hist_input_size,
        stat_input_size,
    ):

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):

                assert (
                    stack_types[i] == "identity"
                ), f"Block type {stack_types[i]} not found!"

                n_theta = input_size + self.outputsize_multiplier * max(
                    h // n_freq_downsample[i], 1
                )
                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=h,
                    out_features=self.outputsize_multiplier,
                    interpolation_mode=interpolation_mode,
                )

                nbeats_block = NHITSBlock(
                    h=h,
                    input_size=input_size,
                    futr_input_size=futr_input_size,
                    hist_input_size=hist_input_size,
                    stat_input_size=stat_input_size,
                    n_theta=n_theta,
                    mlp_units=mlp_units,
                    n_pool_kernel_size=n_pool_kernel_size[i],
                    pooling_mode=pooling_mode,
                    basis=basis,
                    dropout_prob=dropout_prob_theta,
                    activation=activation,
                )

                # Select type of evaluation and apply it to all layers of block
                block_list.append(nbeats_block)

        return block_list

    def forward(self, batch_x):
        batch_x = self.revin(batch_x, 'norm')
        # Parse windows_batch
        insample_y = torch.permute(batch_x, (0,2,1))
        batch_size = insample_y.shape[0]
        enc_in = insample_y.shape[1]
        insample_y = torch.reshape(insample_y, (batch_size*enc_in, -1))
        assert insample_y.shape[1] == self.input_size
        insample_mask = torch.ones_like(insample_y)
        # Parse windows_batch

        futr_exog = None
        hist_exog = None
        stat_exog = None

        # insample
        residuals = insample_y.flip(dims=(-1,))  # backcast init
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:, None]  # Level with Naive1
        block_forecasts = [forecast.repeat(1, self.h, 1)]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

            if self.decompose_forecast:
                block_forecasts.append(block_forecast)

        if self.decompose_forecast:
            # (n_batch, n_blocks, h, output_size)
            block_forecasts = torch.stack(block_forecasts)
            block_forecasts = block_forecasts.permute(1, 0, 2, 3)
            block_forecasts = block_forecasts.squeeze(-1)  # univariate output
            return block_forecasts
        else:
            forecast = torch.reshape(forecast, (batch_size, enc_in, self.h, self.outputsize_multiplier))
            forecast = torch.permute(forecast, (0,2,1,3))
            mu_out = forecast[:,:,:,0]
            std_out = self.std_activ(forecast[:,:,:,1])
            if self.method == 'PROFHIT':
                mu_out, std_out = self.refine(mu_out, std_out)
                mu_2 = torch.einsum('iv ,blv->bli', self.S_mat, mu_out)
                sig_2 = torch.einsum('iv ,blv->bli', self.S_mat, torch.square(std_out))
                JFD = (1/2)*((torch.square(std_out) + torch.square(mu_out - mu_2))/(2*sig_2) + (sig_2 + torch.square(mu_out-mu_2))/(2*torch.square(std_out)) - 1 )
                mu_out = self.revin(mu_out, 'denorm')
                std_out = self.revin(std_out, 'denorm_scale')
                return mu_out, std_out, JFD
            elif self.method == 'HierE2E':
                mu_out = self.revin(mu_out, 'denorm')
                std_out = self.revin(std_out, 'denorm_scale')
                return mu_out, std_out
            elif self.method == 'CLOVER':
                mu_out = self.revin(mu_out, 'denorm')
                std_out = self.revin(std_out, 'denorm_scale')
                factor_vec = forecast[:,:,:,2:]
                factor_vec = factor_vec.reshape(factor_vec.shape[0], -1, self.enc_in).contiguous()
                factor_vec = self.revin(factor_vec, 'denorm_scale')
                factor_vec = factor_vec.reshape(factor_vec.shape[0], self.h,self.enc_in, self.n_components).contiguous()
                return mu_out, std_out, factor_vec
        
class NHITS_HierE2E(HierE2E):
    def __init__(self, seq_len, pred_len, 
                 batch_size, test_batch_size, learning_rate,
                   n_samples, criterion, quantiles, recon, scaler, affine, enc_in, dataset_type, **kwargs
                   ):
        super().__init__(
            batch_size, test_batch_size, learning_rate, 
            n_samples, criterion, quantiles, recon, dataset_type, **kwargs
            )
        self.model = NHITS(h=pred_len,input_size=seq_len, activation='ReLU', scaler=scaler, affine=affine, enc_in=enc_in)

        self.save_hyperparameters()



    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('NBEATS_HierE2E')
        
        ## HierE2E Parameters
        model_parser.add_argument('--learning_rate', type=float,default=0.0001)
        model_parser.add_argument('--criterion', type=str, choices = ['sCRPS','CRPS', 'likelihood'],default='likelihood')
        model_parser.add_argument('--quantiles', type=int,default=9)
        model_parser.add_argument('--n_samples', type=int,default=200)
        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        model_parser.add_argument('--recon', type=str, choices = ['none','BU', 'proj'], default='proj')


        return parent_parser
        
class NHITS_PROFHIT(PROFHIT):
    def __init__(self, seq_len, pred_len, 
                 batch_size, test_batch_size, learning_rate,
                   n_samples, criterion, quantiles, lam, scaler, affine, enc_in, dataset_type, **kwargs
                   ):
        super().__init__(
            batch_size, test_batch_size, learning_rate, 
            n_samples, criterion, quantiles, lam, dataset_type, **kwargs
            )
        self.model = NHITS(h=pred_len,input_size=seq_len, activation='ReLU', scaler=scaler, affine=affine, enc_in=enc_in, method = 'PROFHIT', S_mat = self.SG)

        self.save_hyperparameters()



    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('NBEATS_HierE2E')
        
        ## HierE2E Parameters
        model_parser.add_argument('--learning_rate', type=float,default=0.0001)
        model_parser.add_argument('--criterion', type=str, choices = ['CRPS', 'likelihood'],default='likelihood')
        model_parser.add_argument('--quantiles', type=int,default=9)
        model_parser.add_argument('--n_samples', type=int,default=200)
        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        model_parser.add_argument('--lam', type=float, default=0.5)
        return parent_parser
        
class NHITS_CLOVER(CLOVER):
    def __init__(self, seq_len, pred_len, 
                 batch_size, test_batch_size, learning_rate,
                   n_samples, quantiles, scaler, affine, enc_in, n_components, dataset_type, criterion, **kwargs
                   ):
        super().__init__(
            batch_size, test_batch_size, learning_rate, 
            n_samples, quantiles, n_components, criterion, dataset_type, **kwargs
            )
        self.model = NHITS(h=pred_len,input_size=seq_len, activation='ReLU', scaler=scaler, 
                            affine=affine, enc_in=enc_in - self.n_high, method='CLOVER', n_components=n_components)

        self.save_hyperparameters()



    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('NBEATS_CLOVER')
        
        ## CLOVER Parameters
        model_parser.add_argument('--learning_rate', type=float,default=0.0001)
        model_parser.add_argument('--quantiles', type=int,default=9)
        model_parser.add_argument('--n_samples', type=int,default=200)
        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        model_parser.add_argument('--n_components', type=int, default = 5)
        model_parser.add_argument('--criterion', type=str, default = 'CRPS')


        return parent_parser