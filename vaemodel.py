import copy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
from data_loader import DATA_LOADER as dataloader
import final_classifier as classifier
import models

"""
#vaemodel
"""


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction = nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


class Model(nn.Module):
    def __init__(self, hyperparameters):
        super(Model, self).__init__()
        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources = ['resnet_features', self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = 32
        self.img_seen_samples = hyperparameters['samples_per_class'][self.DATASET][0]
        self.att_seen_samples = hyperparameters['samples_per_class'][self.DATASET][1]
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        self.dataset = dataloader(self.DATASET, copy.deepcopy(self.auxiliary_data_source), device=self.device)
        if self.DATASET == 'CUB':
            self.num_classes = 200
            self.num_novel_classes = 50
        elif self.DATASET == 'SUN':
            self.num_classes = 717
            self.num_novel_classes = 72
        elif self.DATASET == 'AWA1' or self.DATASET == 'AWA2':
            self.num_classes = 50
            self.num_novel_classes = 10
        feature_dimensions = [2048, self.dataset.aux_data.size(1)]

        # Here, the encoders and decoders for all modalities are created and put into dict
        self.encoder = {}
        # for datatype, dim in zip(self.all_data_sources, feature_dimensions):
        #     self.encoder[datatype] = models.encoder_template(dim, self.latent_size, self.hidden_size_rule[datatype], self.device)
        #     print(str(datatype) + ' ' + str(dim))
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.encoder[datatype] = models.encoder_template_disentangle(dim, self.latent_size, self.hidden_size_rule[datatype], self.device)
            print(str(datatype) + ' ' + str(dim))
        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size, dim, self.hidden_size_rule[datatype], self.device)
        # for datatype, dim in zip(self.all_data_sources, feature_dimensions):
        #     self.decoder[datatype] = models.decoder_template(self.latent_size*2, dim, self.hidden_size_rule[datatype], self.device)

        # An optimizer for all encoders and decoders is defined here
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize += list(self.encoder[datatype].parameters())
            parameters_to_optimize += list(self.decoder[datatype].parameters())

        """
        """
        # self.clf_img = LINEAR_LOGSOFTMAX(input_dim=2048, nclass=200).to(self.device)
        # self.clf_img_z = LINEAR_LOGSOFTMAX(input_dim=64, nclass=200).to(self.device)
        # self.clf_att = LINEAR_LOGSOFTMAX(input_dim=312, nclass=200).to(self.device)
        # self.clf_att_z = LINEAR_LOGSOFTMAX(input_dim=64, nclass=200).to(self.device)

        # self.clf_img = LINEAR_LOGSOFTMAX(input_dim=2048, nclass=717).to(self.device)
        # self.clf_img_z = LINEAR_LOGSOFTMAX(input_dim=64, nclass=717).to(self.device)
        # self.clf_att = LINEAR_LOGSOFTMAX(input_dim=102, nclass=717).to(self.device)
        # self.clf_att_z = LINEAR_LOGSOFTMAX(input_dim=64, nclass=717).to(self.device)
        #
        self.clf_img = LINEAR_LOGSOFTMAX(input_dim=2048, nclass=200).to(self.device)
        self.clf_img_z = LINEAR_LOGSOFTMAX(input_dim=64, nclass=200).to(self.device)
        self.clf_att = LINEAR_LOGSOFTMAX(input_dim=312 , nclass=200).to(self.device)
        self.clf_att_z = LINEAR_LOGSOFTMAX(input_dim=64, nclass=200).to(self.device)

        parameters_to_optimize +=list(self.clf_img.parameters())
        parameters_to_optimize +=list(self.clf_att.parameters())
        parameters_to_optimize +=list(self.clf_img_z.parameters())
        parameters_to_optimize +=list(self.clf_att_z.parameters())

        self.optimizer = optim.Adam(parameters_to_optimize, lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        if self.reco_loss_function == 'l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)
        elif self.reco_loss_function == 'l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self, label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label == classes[i]] = i
        return mapped_label

    def trainstep(self, img, att, label):
        """
        ##############################################
        # Encode image features and additional
        # features
        ##############################################
        """
        mu_img_exc, logvar_img_exc, mu_img_dis, logvar_img_dis = self.encoder['resnet_features'](img)
        z_from_img_exc = self.reparameterize(mu_img_exc, logvar_img_exc)
        z_from_img_dis = self.reparameterize(mu_img_dis, logvar_img_dis)
        z_from_img = z_from_img_exc + z_from_img_dis

        mu_att_exc, logvar_att_exc, mu_att_dis, logvar_att_dis = self.encoder[self.auxiliary_data_source](att)
        z_from_att_exc = self.reparameterize(mu_att_exc, logvar_att_exc)
        z_from_att_dis = self.reparameterize(mu_att_dis, logvar_att_dis)
        z_from_att = z_from_att_exc + z_from_att_dis

        """
        shuffle classification loss
        """
        z_from_img_shuffle_exc = z_from_img_exc[torch.randperm(z_from_img_exc.shape[0])]
        z_from_att_shuffle_exc = z_from_att_exc[torch.randperm(z_from_att_exc.shape[0])]
        z_from_img_shuffle = z_from_img_shuffle_exc + z_from_img_dis
        z_from_att_shuffle = z_from_att_shuffle_exc + z_from_att_dis
        img_from_img_shuffle = self.decoder['resnet_features'](z_from_img_shuffle)
        att_from_att_shuffle = self.decoder[self.auxiliary_data_source](z_from_att_shuffle)

        loss_shuffle_classification_img_z = self.clf_img_z.lossfunction(self.clf_img_z(z_from_img_shuffle), label)
        loss_shuffle_classification_att_z = self.clf_att_z.lossfunction(self.clf_att_z(z_from_att_shuffle), label)
        loss_shuffle_classification_img = self.clf_img.lossfunction(self.clf_img(img_from_img_shuffle), label)
        loss_shuffle_classification_att = self.clf_att.lossfunction(self.clf_att(att_from_att_shuffle), label)

        """
        triplet loss
        """
        triplet_loss = nn.TripletMarginLoss(margin=10)
        loss_triplet = triplet_loss(z_from_img, z_from_att, z_from_att[torch.range(z_from_att.shape[0]-1, 0, -1).long()])
        # loss_triplet_dis = triplet_loss(z_from_img_dis, z_from_att_dis, z_from_att_dis[torch.range(z_from_att_dis.shape[0]-1, 0, -1).long()])
        # loss_triplet_exc = triplet_loss(z_from_img_exc, z_from_att_exc, z_from_att_exc[torch.range(z_from_att_exc.shape[0]-1, 0, -1).long()])

        # loss_triplet_dis = triplet_loss(z_from_img_dis, z_from_att_dis, z_from_img_exc)
        # loss_triplet_exc = triplet_loss(z_from_img_exc, z_from_att_exc, z_from_img_dis)
        # loss_triplet_dis = triplet_loss(mu_img_dis, mu_att_dis, mu_att_exc) + triplet_loss(logvar_img_dis.exp(), logvar_att_dis.exp(), logvar_att_exc.exp())
        # loss_triplet_exc = triplet_loss(mu_img_exc, mu_att_exc, mu_att_dis) + triplet_loss(logvar_img_exc.exp(), logvar_att_exc.exp(), logvar_att_dis.exp())
        # loss_triplet_dis = triplet_loss(mu_img_dis, mu_att_dis, mu_img_exc) + triplet_loss(logvar_img_dis.exp(),
        #                                                                                    logvar_att_dis.exp(),
        #                                                                                    logvar_img_exc.exp())
        # loss_triplet_exc = triplet_loss(mu_img_exc, mu_att_exc, mu_img_dis) + triplet_loss(logvar_img_exc.exp(),
        #                                                                                    logvar_att_exc.exp(),
        #                                                                                    logvar_img_dis.exp())

        # loss_triplet_dis = triplet_loss(z_from_img_dis, z_from_att_dis, z_from_att_dis[torch.randperm(z_from_att_dis.shape[0])]) \
        #                    + triplet_loss(z_from_att_dis, z_from_img_dis, z_from_img_dis[torch.randperm(z_from_img_dis.shape[0])])
        # loss_triplet_exc = triplet_loss(z_from_img_exc, z_from_att_exc, z_from_att_exc[torch.randperm(z_from_att_exc.shape[0])])\
        #                    + triplet_loss(z_from_att_exc, z_from_img_exc, z_from_img_exc[torch.randperm(z_from_img_exc.shape[0])])

        """
        ##############################################
        # Reconstruct inputs
        ##############################################
        """
        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)
        reconstruction_loss_img = self.reconstruction_criterion(img_from_img, img)
        reconstruction_loss_att = self.reconstruction_criterion(att_from_att, att)
        # reconstruction_loss = self.reconstruction_criterion(img_from_img, z_from_img_specific_0) \
        #                       + self.reconstruction_criterion(att_from_att, att)
        """
        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        """
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)
        cross_reconstruction_loss_img = self.reconstruction_criterion(img_from_att, img)
        cross_reconstruction_loss_att = self.reconstruction_criterion(att_from_img, att)

        """
        ##############################################
        # KL-Divergence
        ##############################################
        """
        KLD = (0.5 * torch.sum(1 + logvar_att_dis - mu_att_dis.pow(2) - logvar_att_dis.exp())) \
              + (0.5 * torch.sum(1 + logvar_img_dis - mu_img_dis.pow(2) - logvar_img_dis.exp()))
        # KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
        #       + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))
        """
        ##############################################
        # Distribution Alignment
        ##############################################
         """
        distance = torch.sqrt(torch.sum((mu_img_dis - mu_att_dis) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_img_dis.exp()) - torch.sqrt(logvar_att_dis.exp())) ** 2, dim=1))
        distance = distance.sum()
        # distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
        #                       torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))
        # distance = distance.sum()

        """
        ##############################################
        # constraint on exc features
        ##############################################
        """
        KLD_exc = (0.5 * torch.sum(1 + logvar_att_exc - mu_att_exc.pow(2) - logvar_att_exc.exp())) \
                     + (0.5 * torch.sum(1 + logvar_img_exc - mu_img_exc.pow(2) - logvar_img_exc.exp()))
        distance_exc = torch.sqrt(torch.sum((mu_img_exc - mu_att_exc) ** 2, dim=1) + torch.sum(
            (torch.sqrt(logvar_img_exc.exp()) - torch.sqrt(logvar_att_exc.exp())) ** 2, dim=1))
        distance_exc = distance_exc.sum()

        # KLD_exc1 = (0.5 * torch.sum(1 + logvar_att_exc1 - mu_att_exc1.pow(2) - logvar_att_exc1.exp())) \
        #           + (0.5 * torch.sum(1 + logvar_img_exc1 - mu_img_exc1.pow(2) - logvar_img_exc1.exp()))
        # distance_exc1 = torch.sqrt(torch.sum((mu_img_exc1 - mu_att_exc1) ** 2, dim=1) + torch.sum(
        #     (torch.sqrt(logvar_img_exc1.exp()) - torch.sqrt(logvar_att_exc1.exp())) ** 2, dim=1))
        # distance_exc1 = distance_exc1.sum()

        """
        ##############################################
        # scale the loss terms according to the warmup
        # schedule
        ##############################################
         """
        f1 = 1.0*(self.current_epoch - self.warmup['cross_reconstruction']['start_epoch'])/(1.0*(self.warmup['cross_reconstruction']['end_epoch'] - self.warmup['cross_reconstruction']['start_epoch']))
        f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
        cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1, 0), self.warmup['cross_reconstruction']['factor'])])
        f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / (1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])
        f3 = 1.0*(self.current_epoch - self.warmup['distance']['start_epoch'])/(1.0*(self.warmup['distance']['end_epoch'] - self.warmup['distance']['start_epoch']))
        f3 = f3*(1.0*self.warmup['distance']['factor'])
        distance_factor = torch.cuda.FloatTensor([min(max(f3, 0), self.warmup['distance']['factor'])])
        """
        ##############################################
        # Put the loss together and call the optimizer
        ##############################################
         """
        self.optimizer.zero_grad()
        loss = reconstruction_loss_img + reconstruction_loss_att - beta * (1*KLD + 20*KLD_exc)
        loss += loss_shuffle_classification_img_z + loss_shuffle_classification_att_z
        loss += loss_shuffle_classification_img + loss_shuffle_classification_att
        loss += loss_triplet
        if cross_reconstruction_factor > 0:
            loss += cross_reconstruction_factor*(cross_reconstruction_loss_img + cross_reconstruction_loss_att)
        if distance_factor > 0:
            loss += distance_factor * (distance + distance_exc)
            # loss += distance_factor * (loss_triplet_dis + loss_triplet_exc)
            # loss += distance_factor * distance
        loss.backward()
        self.optimizer.step()
        # return loss.item(), reconstruction_loss.item(), KLD.item(), std_commom.item(), loss_classification_img_specific.item(), loss_classification_att_specific.item(), loss_generator.item(), loss_generator_img.item()
        return loss.item(), reconstruction_loss_img.item(), reconstruction_loss_att.item(), KLD.item(), KLD_exc.item(), \
               loss_shuffle_classification_img_z.item(), loss_shuffle_classification_att_z.item(), \
               loss_shuffle_classification_img.item(), loss_shuffle_classification_att.item(),\
               cross_reconstruction_loss_img.item(), cross_reconstruction_loss_att.item(), distance.item(), \
               distance_exc.item()

    def train_vae(self):
        losses = []
        self.dataloader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)  #,num_workers = 4)
        self.dataset.novelclasses =self.dataset.novelclasses.long().cuda()
        self.dataset.seenclasses =self.dataset.seenclasses.long().cuda()
        # leave both statements
        self.train()
        self.reparameterize_with_noise = True
        print('train for reconstruction')
        for epoch in range(0, self.nepoch):
            self.current_epoch = epoch
            i = -1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i += 1
                label, data_from_modalities = self.dataset.next_batch(self.batch_size)
                label = label.long().to(self.device)
                # print(label.unique)
                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(self.device)
                    data_from_modalities[j].requires_grad = False
                # loss, std_loss, generator_loss, clf_img_loss, clf_att_loss = self.trainstep(data_from_modalities[0], data_from_modalities[1], label)
                loss, reconstruction_loss_img, reconstruction_loss_att, KLD_loss, KLD_exc_loss, shuffle_img_z, shuffle_att_z, shuffle_img, shuffle_att, cross_reconstruction_loss_img, cross_reconstruction_loss_att, distance, distance_exc = self.trainstep(data_from_modalities[0], data_from_modalities[1], label)
                # loss, reconstruction_loss_img, reconstruction_loss_att, KLD_loss, KLD_exc_loss, shuffle_img_z, shuffle_att_z, shuffle_img, shuffle_att, cross_reconstruction_loss_img, cross_reconstruction_loss_att, distance, distance_exc, triplet = self.trainstep(data_from_modalities[0], data_from_modalities[1], label)
                if i % 50 == 0:
                    # print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+' | loss ' + str(loss)[:5])
                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+' | loss ' + str(loss)[:5]
                          + '\t' + ' | reconstruction_loss_img ' + str(reconstruction_loss_img)[:5]
                          + '\t' + ' | reconstruction_loss_att ' + str(reconstruction_loss_att)[:5]
                          + '\t' + ' | KLD_loss ' + str(KLD_loss)[:5]
                          + '\t' + ' | KLD_exc_loss ' + str(KLD_exc_loss)[:5]
                          + '\t' + ' | shuffle_img_z_loss ' + str(shuffle_img_z)[:5]
                          + '\t' + ' | shuffle_att_z_loss ' + str(shuffle_att_z)[:5]
                          + '\t' + ' | shuffle_img_loss ' + str(shuffle_img)[:5]
                          + '\t' + ' | shuffle_att_loss ' + str(shuffle_att)[:5]
                          + '\t' + ' | cross_reconstruction_loss_img ' + str(cross_reconstruction_loss_img)[:5]
                          + '\t' + ' | cross_reconstruction_loss_att ' + str(cross_reconstruction_loss_att)[:5]
                          + '\t' + ' | distance ' + str(distance)[:5]
                          + '\t' + ' | distance_exc ' + str(distance_exc)[:5]
                          # + '\t' + ' | triplet ' + str(triplet)[:5]
                          )

                if i % 50 == 0 and i > 0:
                    losses.append(loss)
        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()
        return losses

    def train_classifier(self, show_plots=False):
        if self.num_shots > 0:
            print('================  transfer features from test to train ==================')
            self.dataset.transfer_features(self.num_shots, num_queries='num_features')
        history = []  # stores accuracies
        cls_seenclasses = self.dataset.seenclasses
        cls_novelclasses = self.dataset.novelclasses
        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels']
        novelclass_aux_data = self.dataset.novelclass_aux_data  # access as novelclass_aux_data['resnet_features'], novelclass_aux_data['attributes']
        seenclass_aux_data = self.dataset.seenclass_aux_data
        novel_corresponding_labels = self.dataset.novelclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)
        # The resnet_features for testing the classifier are loaded here
        novel_test_feat = self.dataset.data['test_unseen'][
            'resnet_features']  # self.dataset.test_novel_feature.to(self.device)
        seen_test_feat = self.dataset.data['test_seen'][
            'resnet_features']  # self.dataset.test_seen_feature.to(self.device)
        test_seen_label = self.dataset.data['test_seen']['labels']  # self.dataset.test_seen_label.to(self.device)
        test_novel_label = self.dataset.data['test_unseen']['labels']  # self.dataset.test_novel_label.to(self.device)
        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']
        # in ZSL mode:
        if self.generalized == False:
            # there are only 50 classes in ZSL (for CUB)
            # novel_corresponding_labels =list of all novel classes (as tensor)
            # test_novel_label = mapped to 0-49 in classifier function
            # those are used as targets, they have to be mapped to 0-49 right here:
            novel_corresponding_labels = self.map_label(novel_corresponding_labels, novel_corresponding_labels)
            if self.num_shots > 0:
                # not generalized and at least 1 shot means normal FSL setting (use only unseen classes)
                train_unseen_label = self.map_label(train_unseen_label, cls_novelclasses)
            # for FSL, we train_seen contains the unseen class examples
            # for ZSL, train seen label is not used
            # if self.num_shots>0:
            #    train_seen_label = self.map_label(train_seen_label,cls_novelclasses)
            test_novel_label = self.map_label(test_novel_label, cls_novelclasses)
            # map cls novelclasses last
            cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)
        if self.generalized:
            print('mode: gzsl')
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
            # clf = LINEAR_LOGSOFTMAX(self.latent_size*2, self.num_classes)
        else:
            print('mode: zsl')
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_novel_classes)
        clf.apply(models.weights_init)
        with torch.no_grad():
            ####################################
            # preparing the test set
            # convert raw test data into z vectors
            ####################################
            self.reparameterize_with_noise = False

            # _, _, mu_dis1, var_dis1 = self.encoder['resnet_features'](novel_test_feat)
            # test_novel_X_dis = self.reparameterize(mu_dis1, var_dis1).to(self.device).data
            # test_novel_X = test_novel_X_dis
            # test_novel_Y = test_novel_label.to(self.device)
            #
            # _, _, mu_dis2, var_dis2 = self.encoder['resnet_features'](seen_test_feat)
            # test_seen_X_dis = self.reparameterize(mu_dis2, var_dis2).to(self.device).data
            # test_seen_X = test_seen_X_dis
            # test_seen_Y = test_seen_label.to(self.device)
            mu_exc1, var_exc1, mu_dis1, var_dis1 = self.encoder['resnet_features'](novel_test_feat)
            test_novel_X_dis = self.reparameterize(mu_dis1, var_dis1).to(self.device).data
            test_novel_X_exc = self.reparameterize(mu_exc1, var_exc1).to(self.device).data
            test_novel_X_exc = test_novel_X_exc[torch.randperm(test_novel_X_exc.shape[0])]
            # test_novel_X = test_novel_X_exc + test_novel_X_dis
            # test_novel_X = test_novel_X_exc
            test_novel_X = test_novel_X_dis
            test_novel_Y = test_novel_label.to(self.device)

            mu_exc2, var_exc2, mu_dis2, var_dis2 = self.encoder['resnet_features'](seen_test_feat)
            test_seen_X_exc = self.reparameterize(mu_exc2, var_exc2).to(self.device).data
            test_seen_X_exc = test_seen_X_exc[torch.randperm(test_seen_X_exc.shape[0])]
            test_seen_X_dis = self.reparameterize(mu_dis2, var_dis2).to(self.device).data
            # test_seen_X = test_seen_X_exc + test_seen_X_dis
            # test_seen_X = test_seen_X_exc
            test_seen_X = test_seen_X_dis
            test_seen_Y = test_seen_label.to(self.device)
            ####################################
            # preparing the train set:
            # chose n random image features per
            # class. If n exceeds the number of
            # image features per class, duplicate
            # some. Next, convert them to
            # latent z features.
            ####################################
            self.reparameterize_with_noise = True

            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)
                if sample_per_class != 0 and len(label) != 0:
                    classes = label.unique()
                    for i, s in enumerate(classes):
                        features_of_that_class = features[label == s, :]  # order of features and labels must coincide
                        # if number of selected features is smaller than the number of features we want per class:
                        multiplier = torch.ceil(torch.cuda.FloatTensor(
                            [max(1, sample_per_class / features_of_that_class.size(0))])).long().item()
                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)
                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat(
                                (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),
                                                         dim=0)
                    return features_to_return, labels_to_return
                else:
                    return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])
            # some of the following might be empty tensors if the specified number of
            # samples is zero :
            img_seen_feat, img_seen_label = sample_train_data_on_sample_per_class_basis(
                train_seen_feat, train_seen_label, self.img_seen_samples)
            img_unseen_feat, img_unseen_label = sample_train_data_on_sample_per_class_basis(
                train_unseen_feat, train_unseen_label, self.img_unseen_samples)
            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(
                novelclass_aux_data, novel_corresponding_labels, self.att_unseen_samples)
            att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(
                seenclass_aux_data, seen_corresponding_labels, self.att_seen_samples)

            # def convert_datapoints_to_z(features, encoder):
            #     if features.size(0) != 0:
            #         _, _, mu_dis, logvar_dis = encoder(features)
            #         z_dis = self.reparameterize(mu_dis, logvar_dis)
            #         z = z_dis
            #         return z
            #     else:
            #         return torch.cuda.FloatTensor([])
            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    mu_exc, logvar_exc, mu_dis, logvar_dis = encoder(features)
                    z_exc = self.reparameterize(mu_exc, logvar_exc)
                    z_dis = self.reparameterize(mu_dis, logvar_dis)
                    z_exc = z_exc[torch.randperm(z_exc.shape[0])]
                    # z = z_exc + z_dis
                    # z = z_exc
                    z = z_dis
                    return z
                else:
                    return torch.cuda.FloatTensor([])
            z_seen_img = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
            print(z_seen_img.size())
            z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])
            print(z_unseen_img.size())
            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
            print(z_seen_att.size())
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])
            print(z_unseen_att.size())
            train_Z = [z_seen_img, z_unseen_img, z_seen_att, z_unseen_att]
            train_L = [img_seen_label, img_unseen_label, att_seen_label, att_unseen_label]
            # empty tensors are sorted out
            train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
            train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]
            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)

            # np.save('/home/xzhe/Projects/CADA-VAE-PyTorch-master/train_x_dis.npy', train_X.cpu().numpy())
            # np.save('/home/xzhe/Projects/CADA-VAE-PyTorch-master/train_y.npy', train_Y.cpu().numpy())

        ############################################################
        # initializing the classifier and train one epoch
        ############################################################
        cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X, test_novel_Y,
                                    cls_seenclasses, cls_novelclasses,
                                    self.num_classes, self.device, self.lr_cls, 0.5, 1,
                                    self.classifier_batch_size,
                                    self.generalized)
        for k in range(self.cls_train_epochs):
            if k > 0:
                if self.generalized:
                    cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
                else:
                    cls.acc = cls.fit_zsl()
            if self.generalized:
                print('[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (
                k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))
                history.append([torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(),
                                torch.tensor(cls.H).item()])
            else:
                print('[%.1f]  acc=%.4f ' % (k, cls.acc))
                history.append([0, torch.tensor(cls.acc).item(), 0])
        if self.generalized:
            return torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(cls.H).item(), history
        else:
            return 0, torch.tensor(cls.acc).item(), 0, history


# class Model(nn.Module):
#     def __init__(self, hyperparameters):
#         super(Model, self).__init__()
#         self.device = hyperparameters['device']
#         self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
#         self.all_data_sources = ['resnet_features', self.auxiliary_data_source]
#         self.DATASET = hyperparameters['dataset']
#         self.num_shots = hyperparameters['num_shots']
#         self.latent_size = hyperparameters['latent_size']
#         self.batch_size = hyperparameters['batch_size']
#         self.hidden_size_rule = hyperparameters['hidden_size_rule']
#         self.warmup = hyperparameters['model_specifics']['warmup']
#         self.generalized = hyperparameters['generalized']
#         self.classifier_batch_size = 32
#         self.img_seen_samples = hyperparameters['samples_per_class'][self.DATASET][0]
#         self.att_seen_samples = hyperparameters['samples_per_class'][self.DATASET][1]
#         self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
#         self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
#         self.reco_loss_function = hyperparameters['loss']
#         self.nepoch = hyperparameters['epochs']
#         self.lr_cls = hyperparameters['lr_cls']
#         self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
#         self.cls_train_epochs = hyperparameters['cls_train_steps']
#         self.dataset = dataloader(self.DATASET, copy.deepcopy(self.auxiliary_data_source), device=self.device)
#         if self.DATASET == 'CUB':
#             self.num_classes = 200
#             self.num_novel_classes = 50
#         elif self.DATASET == 'SUN':
#             self.num_classes = 717
#             self.num_novel_classes = 72
#         elif self.DATASET == 'AWA1' or self.DATASET == 'AWA2':
#             self.num_classes = 50
#             self.num_novel_classes = 10
#         feature_dimensions = [2048, self.dataset.aux_data.size(1)]
#         # Here, the encoders and decoders for all modalities are created and put into dict
#         self.encoder = {}
#         # for datatype, dim in zip(self.all_data_sources, feature_dimensions):
#         #     self.encoder[datatype] = models.encoder_template(dim, self.latent_size, self.hidden_size_rule[datatype], self.device)
#         #     print(str(datatype) + ' ' + str(dim))
#         for datatype, dim in zip(self.all_data_sources, feature_dimensions):
#             self.encoder[datatype] = models.encoder_template_disentangle(dim, self.latent_size, self.hidden_size_rule[datatype], self.device)
#             print(str(datatype) + ' ' + str(dim))
#         self.decoder = {}
#         # for datatype, dim in zip(self.all_data_sources, feature_dimensions):
#         #     self.decoder[datatype] = models.decoder_template(self.latent_size, dim, self.hidden_size_rule[datatype], self.device)
#         # for datatype, dim in zip(self.all_data_sources, feature_dimensions):
#         #     self.decoder[datatype] = models.decoder_template(self.latent_size*2, dim, self.hidden_size_rule[datatype], self.device)
#         """
#         """
#         self.decoder['resnet_features'] = models.decoder_template(self.latent_size*2, 2048, self.hidden_size_rule['resnet_features'], self.device)
#         self.decoder[self.auxiliary_data_source] = models.decoder_template(self.latent_size, 102, self.hidden_size_rule[self.auxiliary_data_source], self.device)
#
#
#         # An optimizer for all encoders and decoders is defined here
#         parameters_to_optimize = list(self.parameters())
#         for datatype in self.all_data_sources:
#             parameters_to_optimize += list(self.encoder[datatype].parameters())
#             parameters_to_optimize += list(self.decoder[datatype].parameters())
#
#         """
#         """
#         # self.clf_img = LINEAR_LOGSOFTMAX(input_dim=2048, nclass=200).to(self.device)
#         # self.clf_img_z = LINEAR_LOGSOFTMAX(input_dim=128, nclass=200).to(self.device)
#
#         self.clf_img = LINEAR_LOGSOFTMAX(input_dim=2048, nclass=717).to(self.device)
#         self.clf_img_z = LINEAR_LOGSOFTMAX(input_dim=128, nclass=717).to(self.device)
#
#         parameters_to_optimize +=list(self.clf_img.parameters())
#         parameters_to_optimize +=list(self.clf_img_z.parameters())
#
#         self.optimizer = optim.Adam(parameters_to_optimize, lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
#         if self.reco_loss_function == 'l2':
#             self.reconstruction_criterion = nn.MSELoss(size_average=False)
#         elif self.reco_loss_function == 'l1':
#             self.reconstruction_criterion = nn.L1Loss(size_average=False)
#
#     def reparameterize(self, mu, logvar):
#         if self.reparameterize_with_noise:
#             sigma = torch.exp(logvar)
#             eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
#             eps = eps.expand(sigma.size())
#             return mu + sigma*eps
#         else:
#             return mu
#
#     def forward(self):
#         pass
#
#     def map_label(self, label, classes):
#         mapped_label = torch.LongTensor(label.size()).to(self.device)
#         for i in range(classes.size(0)):
#             mapped_label[label == classes[i]] = i
#         return mapped_label
#
#     def trainstep(self, img, att, label):
#         """
#         ##############################################
#         # Encode image features and additional
#         # features
#         ##############################################
#         """
#         mu_img_exc, logvar_img_exc, mu_img_dis, logvar_img_dis = self.encoder['resnet_features'](img)
#         z_from_img_exc = self.reparameterize(mu_img_exc, logvar_img_exc)
#         z_from_img_dis = self.reparameterize(mu_img_dis, logvar_img_dis)
#         # z_from_img = z_from_img_exc + z_from_img_dis
#         z_from_img = torch.cat((z_from_img_exc, z_from_img_dis), 1)
#
#         _, _, mu_att_dis, logvar_att_dis = self.encoder[self.auxiliary_data_source](att)
#         z_from_att_dis = self.reparameterize(mu_att_dis, logvar_att_dis)
#         # z_from_att = z_from_img_exc + z_from_att_dis
#         z_from_att = torch.cat((z_from_img_exc, z_from_att_dis), 1)
#
#         """
#         shuffle classification loss
#         """
#         z_from_img_shuffle_exc = z_from_img_exc[torch.randperm(z_from_img_exc.shape[0])]
#         # z_from_img_shuffle = z_from_img_shuffle_exc + z_from_img_dis
#         z_from_img_shuffle = torch.cat((z_from_img_shuffle_exc, z_from_img_dis), 1)
#         img_from_img_shuffle = self.decoder['resnet_features'](z_from_img_shuffle)
#
#         loss_shuffle_classification_img_z = self.clf_img_z.lossfunction(self.clf_img_z(z_from_img_shuffle), label)
#         loss_shuffle_classification_img = self.clf_img.lossfunction(self.clf_img(img_from_img_shuffle), label)
#
#         """
#         triplet loss
#         """
#         triplet_loss = nn.TripletMarginLoss(margin=10)
#         loss_triplet = triplet_loss(z_from_img, z_from_att, z_from_att[torch.range(z_from_att.shape[0]-1, 0, -1).long()])
#         # loss_triplet_dis = triplet_loss(z_from_img_dis, z_from_att_dis, z_from_att_dis[torch.range(z_from_att_dis.shape[0]-1, 0, -1).long()])
#         # loss_triplet_dis = triplet_loss(z_from_att_dis, z_from_img_dis, z_from_img_exc)
#         loss_triplet_dis = triplet_loss(mu_att_dis, mu_img_dis, mu_img_exc) + triplet_loss(logvar_att_dis.exp(), logvar_img_dis.exp(), logvar_img_exc.exp())
#
#         """
#         ##############################################
#         # Reconstruct inputs
#         ##############################################
#         """
#         img_from_img = self.decoder['resnet_features'](z_from_img)
#         att_from_att = self.decoder[self.auxiliary_data_source](z_from_att_dis)
#         reconstruction_loss_img = self.reconstruction_criterion(img_from_img, img)
#         reconstruction_loss_att = self.reconstruction_criterion(att_from_att, att)
#         # reconstruction_loss = self.reconstruction_criterion(img_from_img, z_from_img_specific_0) \
#         #                       + self.reconstruction_criterion(att_from_att, att)
#         """
#         ##############################################
#         # Cross Reconstruction Loss
#         ##############################################
#         """
#         img_from_att = self.decoder['resnet_features'](z_from_att)
#         att_from_img = self.decoder[self.auxiliary_data_source](z_from_img_dis)
#         cross_reconstruction_loss_img = self.reconstruction_criterion(img_from_att, img)
#         cross_reconstruction_loss_att = self.reconstruction_criterion(att_from_img, att)
#
#         """
#         ##############################################
#         # KL-Divergence
#         ##############################################
#         """
#         KLD = (0.5 * torch.sum(1 + logvar_att_dis - mu_att_dis.pow(2) - logvar_att_dis.exp())) \
#               + (0.5 * torch.sum(1 + logvar_img_dis - mu_img_dis.pow(2) - logvar_img_dis.exp()))
#         # KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
#         #       + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))
#         """
#         ##############################################
#         # Distribution Alignment
#         ##############################################
#          """
#         distance = torch.sqrt(torch.sum((mu_img_dis - mu_att_dis) ** 2, dim=1) + \
#                               torch.sum((torch.sqrt(logvar_img_dis.exp()) - torch.sqrt(logvar_att_dis.exp())) ** 2, dim=1))
#         distance = distance.sum()
#         # distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
#         #                       torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))
#         # distance = distance.sum()
#
#         """
#         ##############################################
#         # constraint on exc features
#         ##############################################
#         """
#         KLD_exc = (0.5 * torch.sum(1 + logvar_img_exc - mu_img_exc.pow(2) - logvar_img_exc.exp()))
#
#         """
#         ##############################################
#         # scale the loss terms according to the warmup
#         # schedule
#         ##############################################
#          """
#         f1 = 1.0*(self.current_epoch - self.warmup['cross_reconstruction']['start_epoch'])/(1.0*(self.warmup['cross_reconstruction']['end_epoch'] - self.warmup['cross_reconstruction']['start_epoch']))
#         f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
#         cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1, 0), self.warmup['cross_reconstruction']['factor'])])
#         f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / (1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
#         f2 = f2 * (1.0 * self.warmup['beta']['factor'])
#         beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])
#         f3 = 1.0*(self.current_epoch - self.warmup['distance']['start_epoch'])/(1.0*(self.warmup['distance']['end_epoch'] - self.warmup['distance']['start_epoch']))
#         f3 = f3*(1.0*self.warmup['distance']['factor'])
#         distance_factor = torch.cuda.FloatTensor([min(max(f3, 0), self.warmup['distance']['factor'])])
#         """
#         ##############################################
#         # Put the loss together and call the optimizer
#         ##############################################
#          """
#         self.optimizer.zero_grad()
#         # loss = reconstruction_loss - beta * KLD
#         loss = 0.9*reconstruction_loss_img + 0.5*reconstruction_loss_att - beta * (20*KLD + 200*KLD_exc)
#         loss += loss_shuffle_classification_img_z*95
#         loss += loss_shuffle_classification_img*110
#         # loss += loss_triplet
#         if cross_reconstruction_factor > 0:
#             loss += cross_reconstruction_factor*(0.05*cross_reconstruction_loss_img + 0.4*cross_reconstruction_loss_att)
#             # loss += loss_shuffle_classification_img_z + loss_shuffle_classification_att_z
#             # loss += loss_shuffle_classification_img + loss_shuffle_classification_att
#         if distance_factor > 0:
#             # loss += distance_factor * distance
#             loss += distance_factor * loss_triplet_dis*20
#         loss.backward()
#         self.optimizer.step()
#         # return loss.item(), reconstruction_loss.item(), KLD.item(), std_commom.item(), loss_classification_img_specific.item(), loss_classification_att_specific.item(), loss_generator.item(), loss_generator_img.item()
#         return loss.item(), reconstruction_loss_img.item(), reconstruction_loss_att.item(), KLD.item(), KLD_exc.item(), \
#                loss_shuffle_classification_img_z.item(), cross_reconstruction_loss_img.item(), cross_reconstruction_loss_att.item(), \
#                distance.item(), loss_triplet_dis.item()
#
#     def train_vae(self):
#         losses = []
#         self.dataloader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)  #,num_workers = 4)
#         self.dataset.novelclasses =self.dataset.novelclasses.long().cuda()
#         self.dataset.seenclasses =self.dataset.seenclasses.long().cuda()
#         # leave both statements
#         self.train()
#         self.reparameterize_with_noise = True
#         print('train for reconstruction')
#         for epoch in range(0, self.nepoch):
#             self.current_epoch = epoch
#             i = -1
#             for iters in range(0, self.dataset.ntrain, self.batch_size):
#                 i += 1
#                 label, data_from_modalities = self.dataset.next_batch(self.batch_size)
#                 label = label.long().to(self.device)
#                 # print(label.unique)
#                 for j in range(len(data_from_modalities)):
#                     data_from_modalities[j] = data_from_modalities[j].to(self.device)
#                     data_from_modalities[j].requires_grad = False
#                 # loss, std_loss, generator_loss, clf_img_loss, clf_att_loss = self.trainstep(data_from_modalities[0], data_from_modalities[1], label)
#                 loss, reconstruction_loss_img, reconstruction_loss_att, KLD_loss, KLD_exc_loss, shuffle_img_z, cross_reconstruction_loss_img, cross_reconstruction_loss_att, distance, triplet = self.trainstep(data_from_modalities[0], data_from_modalities[1], label)
#                 if i % 50 == 0:
#                     # print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+' | loss ' + str(loss)[:5])
#                     print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+' | loss ' + str(loss)[:5]
#                           + '\t' + ' | reconstruction_loss_img ' + str(reconstruction_loss_img)[:5]
#                           + '\t' + ' | reconstruction_loss_att ' + str(reconstruction_loss_att)[:5]
#                           + '\t' + ' | KLD_loss ' + str(KLD_loss)[:5]
#                           + '\t' + ' | KLD_exc_loss ' + str(KLD_exc_loss)[:5]
#                           + '\t' + ' | shuffle_img_z_loss ' + str(shuffle_img_z)[:5]
#                           + '\t' + ' | cross_reconstruction_loss_img ' + str(cross_reconstruction_loss_img)[:5]
#                           + '\t' + ' | cross_reconstruction_loss_att ' + str(cross_reconstruction_loss_att)[:5]
#                           + '\t' + ' | distance ' + str(distance)[:5]
#                           + '\t' + ' | triplet ' + str(triplet)[:5]
#                           )
#
#                 if i % 50 == 0 and i > 0:
#                     losses.append(loss)
#         # turn into evaluation mode:
#         for key, value in self.encoder.items():
#             self.encoder[key].eval()
#         for key, value in self.decoder.items():
#             self.decoder[key].eval()
#         return losses
#
#     def train_classifier(self, show_plots=False):
#         if self.num_shots > 0:
#             print('================  transfer features from test to train ==================')
#             self.dataset.transfer_features(self.num_shots, num_queries='num_features')
#         history = []  # stores accuracies
#         cls_seenclasses = self.dataset.seenclasses
#         cls_novelclasses = self.dataset.novelclasses
#         train_seen_feat = self.dataset.data['train_seen']['resnet_features']
#         train_seen_label = self.dataset.data['train_seen']['labels']
#         novelclass_aux_data = self.dataset.novelclass_aux_data  # access as novelclass_aux_data['resnet_features'], novelclass_aux_data['attributes']
#         seenclass_aux_data = self.dataset.seenclass_aux_data
#         novel_corresponding_labels = self.dataset.novelclasses.long().to(self.device)
#         seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)
#         # The resnet_features for testing the classifier are loaded here
#         novel_test_feat = self.dataset.data['test_unseen'][
#             'resnet_features']  # self.dataset.test_novel_feature.to(self.device)
#         seen_test_feat = self.dataset.data['test_seen'][
#             'resnet_features']  # self.dataset.test_seen_feature.to(self.device)
#         test_seen_label = self.dataset.data['test_seen']['labels']  # self.dataset.test_seen_label.to(self.device)
#         test_novel_label = self.dataset.data['test_unseen']['labels']  # self.dataset.test_novel_label.to(self.device)
#         train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
#         train_unseen_label = self.dataset.data['train_unseen']['labels']
#         # in ZSL mode:
#         if self.generalized == False:
#             # there are only 50 classes in ZSL (for CUB)
#             # novel_corresponding_labels =list of all novel classes (as tensor)
#             # test_novel_label = mapped to 0-49 in classifier function
#             # those are used as targets, they have to be mapped to 0-49 right here:
#             novel_corresponding_labels = self.map_label(novel_corresponding_labels, novel_corresponding_labels)
#             if self.num_shots > 0:
#                 # not generalized and at least 1 shot means normal FSL setting (use only unseen classes)
#                 train_unseen_label = self.map_label(train_unseen_label, cls_novelclasses)
#             # for FSL, we train_seen contains the unseen class examples
#             # for ZSL, train seen label is not used
#             # if self.num_shots>0:
#             #    train_seen_label = self.map_label(train_seen_label,cls_novelclasses)
#             test_novel_label = self.map_label(test_novel_label, cls_novelclasses)
#             # map cls novelclasses last
#             cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)
#         if self.generalized:
#             print('mode: gzsl')
#             clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
#             # clf = LINEAR_LOGSOFTMAX(self.latent_size*2, self.num_classes)
#         else:
#             print('mode: zsl')
#             clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_novel_classes)
#         clf.apply(models.weights_init)
#         with torch.no_grad():
#             ####################################
#             # preparing the test set
#             # convert raw test data into z vectors
#             ####################################
#             self.reparameterize_with_noise = False
#
#             _, _, mu_dis1, var_dis1 = self.encoder['resnet_features'](novel_test_feat)
#             test_novel_X_dis = self.reparameterize(mu_dis1, var_dis1).to(self.device).data
#             test_novel_X = test_novel_X_dis
#             test_novel_Y = test_novel_label.to(self.device)
#
#             _, _, mu_dis2, var_dis2 = self.encoder['resnet_features'](seen_test_feat)
#             test_seen_X_dis = self.reparameterize(mu_dis2, var_dis2).to(self.device).data
#             test_seen_X = test_seen_X_dis
#             test_seen_Y = test_seen_label.to(self.device)
#             ####################################
#             # preparing the train set:
#             # chose n random image features per
#             # class. If n exceeds the number of
#             # image features per class, duplicate
#             # some. Next, convert them to
#             # latent z features.
#             ####################################
#             self.reparameterize_with_noise = True
#
#             def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
#                 sample_per_class = int(sample_per_class)
#                 if sample_per_class != 0 and len(label) != 0:
#                     classes = label.unique()
#                     for i, s in enumerate(classes):
#                         features_of_that_class = features[label == s, :]  # order of features and labels must coincide
#                         # if number of selected features is smaller than the number of features we want per class:
#                         multiplier = torch.ceil(torch.cuda.FloatTensor(
#                             [max(1, sample_per_class / features_of_that_class.size(0))])).long().item()
#                         features_of_that_class = features_of_that_class.repeat(multiplier, 1)
#                         if i == 0:
#                             features_to_return = features_of_that_class[:sample_per_class, :]
#                             labels_to_return = s.repeat(sample_per_class)
#                         else:
#                             features_to_return = torch.cat(
#                                 (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
#                             labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),
#                                                          dim=0)
#                     return features_to_return, labels_to_return
#                 else:
#                     return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])
#             # some of the following might be empty tensors if the specified number of
#             # samples is zero :
#             img_seen_feat, img_seen_label = sample_train_data_on_sample_per_class_basis(
#                 train_seen_feat, train_seen_label, self.img_seen_samples)
#             img_unseen_feat, img_unseen_label = sample_train_data_on_sample_per_class_basis(
#                 train_unseen_feat, train_unseen_label, self.img_unseen_samples)
#             att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(
#                 novelclass_aux_data, novel_corresponding_labels, self.att_unseen_samples)
#             att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(
#                 seenclass_aux_data, seen_corresponding_labels, self.att_seen_samples)
#
#             def convert_datapoints_to_z(features, encoder):
#                 if features.size(0) != 0:
#                     _, _, mu_dis, logvar_dis = encoder(features)
#                     z_dis = self.reparameterize(mu_dis, logvar_dis)
#                     z = z_dis
#                     return z
#                 else:
#                     return torch.cuda.FloatTensor([])
#             z_seen_img = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
#             print(z_seen_img.size())
#             z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])
#             print(z_unseen_img.size())
#             z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
#             print(z_seen_att.size())
#             z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])
#             print(z_unseen_att.size())
#             train_Z = [z_seen_img, z_unseen_img, z_seen_att, z_unseen_att]
#             train_L = [img_seen_label, img_unseen_label, att_seen_label, att_unseen_label]
#             # empty tensors are sorted out
#             train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
#             train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]
#             train_X = torch.cat(train_X, dim=0)
#             train_Y = torch.cat(train_Y, dim=0)
#         ############################################################
#         # initializing the classifier and train one epoch
#         ############################################################
#         cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X, test_novel_Y,
#                                     cls_seenclasses, cls_novelclasses,
#                                     self.num_classes, self.device, self.lr_cls, 0.5, 1,
#                                     self.classifier_batch_size,
#                                     self.generalized)
#         for k in range(self.cls_train_epochs):
#             if k > 0:
#                 if self.generalized:
#                     cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
#                 else:
#                     cls.acc = cls.fit_zsl()
#             if self.generalized:
#                 print('[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (
#                 k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))
#                 history.append([torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(),
#                                 torch.tensor(cls.H).item()])
#             else:
#                 print('[%.1f]  acc=%.4f ' % (k, cls.acc))
#                 history.append([0, torch.tensor(cls.acc).item(), 0])
#         if self.generalized:
#             return torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(cls.H).item(), history
#         else:
#             return 0, torch.tensor(cls.acc).item(), 0, history
