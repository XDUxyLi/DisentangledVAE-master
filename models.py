import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.io as sio


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# class encoder_template(nn.Module):
#     def __init__(self, input_dim, latent_size, hidden_size_rule, device):
#         super(encoder_template, self).__init__()
#         if len(hidden_size_rule) == 2:
#             self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
#         elif len(hidden_size_rule) == 3:
#             self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1], latent_size]
#         modules = []
#         for i in range(len(self.layer_sizes)-2):
#             modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
#             modules.append(nn.ReLU())
#         self.feature_encoder = nn.Sequential(*modules)
#         self._mu = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self._logvar = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self.apply(weights_init)
#         self.to(device)
#
#     def forward(self, x):
#         # print(x.shape)
#         h = self.feature_encoder(x)
#         mu = self._mu(h)
#         logvar = self._logvar(h)
#         return mu, logvar


# class encoder_template_disentangle(nn.Module):
#     def __init__(self, input_dim, latent_size, hidden_size_rule, device):
#         super(encoder_template_disentangle, self).__init__()
#         if len(hidden_size_rule) == 2:
#             self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
#         elif len(hidden_size_rule) == 3:
#             self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1], latent_size]
#         elif len(hidden_size_rule) == 4:
#             self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1], hidden_size_rule[2], latent_size]
#         modules = []
#         for i in range(len(self.layer_sizes)-2):
#             modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
#             modules.append(nn.ReLU())
#         self.feature_encoder = nn.Sequential(*modules)
#         self._mu_common = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self._logvar_common = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self._mu_specific = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self._logvar_specific = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self.apply(weights_init)
#         self.to(device)
#
#     def forward(self, x):
#         # print(x.shape)
#         h = self.feature_encoder(x)
#         mu_common = self._mu_common(h)
#         logvar_common = self._logvar_common(h)
#         mu_specific = self._mu_specific(h)
#         logvar_specific = self._logvar_specific(h)
#         return mu_common, logvar_common, mu_specific, logvar_specific


class encoder_template_disentangle(nn.Module):
    def __init__(self, input_dim, latent_size, hidden_size_rule, device):
        super(encoder_template_disentangle, self).__init__()
        if len(hidden_size_rule) == 2:
            self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
        elif len(hidden_size_rule) == 3:
            self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1], latent_size]
        elif len(hidden_size_rule) == 4:
            self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1], hidden_size_rule[2], latent_size]
        modules = []
        for i in range(len(self.layer_sizes)-2):
            modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
            modules.append(nn.ReLU())
        self.feature_encoder = nn.Sequential(*modules)
        self._mu_excluded = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self._logvar_excluded = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self._mu_discriminative = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self._logvar_discriminative = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self.apply(weights_init)
        self.to(device)

    def forward(self, x):
        # print(x.shape)
        h = self.feature_encoder(x)
        mu_excluded = self._mu_excluded(h)
        logvar_excluded = self._logvar_excluded(h)
        mu_discriminative = self._mu_discriminative(h)
        logvar_discriminative = self._logvar_discriminative(h)
        return mu_excluded, logvar_excluded, mu_discriminative, logvar_discriminative


# class encoder_template_disentangle(nn.Module):
#     def __init__(self, input_dim, latent_size, hidden_size_rule, device):
#         super(encoder_template_disentangle, self).__init__()
#         if len(hidden_size_rule) == 2:
#             self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
#         elif len(hidden_size_rule) == 3:
#             self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1], latent_size]
#         modules = []
#         for i in range(len(self.layer_sizes)-2):
#             modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
#             modules.append(nn.ReLU())
#         self.feature_encoder_common = nn.Sequential(*modules)
#         self.feature_encoder_specific = nn.Sequential(*modules)
#         self._mu_common = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self._logvar_common = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self._mu_specific = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self._logvar_specific = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self.apply(weights_init)
#         self.to(device)
#
#     def forward(self, x):
#         # print(x.shape)
#         h_common = self.feature_encoder_common(x)
#         h_specific = self.feature_encoder_specific(x)
#         mu_common = self._mu_common(h_common)
#         logvar_common = self._logvar_common(h_common)
#         mu_specific = self._mu_specific(h_specific)
#         logvar_specific = self._logvar_specific(h_specific)
#         return mu_common, logvar_common, mu_specific, logvar_specific


# class encoder_template_1(nn.Module):
#     def __init__(self, input_dim, latent_size, hidden_size_rule, device):
#         super(encoder_template_1, self).__init__()
#         if len(hidden_size_rule) == 2:
#             self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
#         elif len(hidden_size_rule) == 3:
#             self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1], latent_size]
#         modules = []
#         for i in range(len(self.layer_sizes)-2):
#             modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
#             modules.append(nn.ReLU())
#         self.feature_encoder = nn.Sequential(*modules)
#         self._mu = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self._logvar = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
#         self.apply(weights_init)
#         self.to(device)
#         self.transformer = TransformerEncoderLayer(d_model=input_dim, nhead=4)
#
#     def forward(self, x):
#         x1 = self.transformer(x.unsqueeze(0))
#         h = self.feature_encoder(x1.squeeze(0))
#         mu = self._mu(h)
#         logvar = self._logvar(h)
#         return mu, logvar


class decoder_template(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size_rule, device):
        super(decoder_template, self).__init__()
        self.layer_sizes = [input_dim, hidden_size_rule[-1], output_dim]
        self.feature_decoder = nn.Sequential(nn.Linear(input_dim, self.layer_sizes[1]), nn.ReLU(), nn.Linear(self.layer_sizes[1], output_dim))
        self.apply(weights_init)
        self.to(device)

    def forward(self, x):
        return self.feature_decoder(x)


# class decoder_template(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_size_rule, device):
#         super(decoder_template, self).__init__()
#         self.layer_sizes = [input_dim, hidden_size_rule[-1], output_dim]
#         self.feature_decoder_excluded = nn.Sequential(nn.Linear(input_dim, self.layer_sizes[1]), nn.ReLU(), nn.Linear(self.layer_sizes[1], output_dim))
#         self.feature_decoder_discriminative = nn.Sequential(nn.Linear(input_dim, self.layer_sizes[1]), nn.ReLU(), nn.Linear(self.layer_sizes[1], output_dim))
#         self.apply(weights_init)
#         self.to(device)
#
#     def forward(self, x_exc, x_dis):
#         return self.feature_decoder_excluded(x_exc) + self.feature_decoder_discriminative(x_dis)
