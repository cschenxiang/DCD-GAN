import itertools
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from util.image_pool import ImagePool
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class DCDModel(BaseModel):
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for DCD-GAN """
        parser.add_argument('--DCD_mode', type=str, default="DCD", choices='DCD')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_IDT', type=float, default=1.0, help='weight for l1 identical loss: (G(X),X)')
        parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')
        parser.add_argument('--cyc_L1', type=float, default=1.0, help='weight for L1 loss')
        parser.add_argument('--lambda_feat', type=float, default=1.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_identity', type=float, default=1.0, help='weight for identity loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization.")

        parser.set_defaults(pool_size=0)

        opt, _ = parser.parse_known_args()

        if opt.DCD_mode.lower() == "dcd":
            parser.set_defaults(nce_idt=True, lambda_NCE=2.0)
        else:
            raise ValueError(opt.DCD_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['D_A', 'G_A', 'NCE1', 'D_B', 'G_B', 'NCE2', 'G']
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['idt_B', 'idt_A']
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B

        if self.isTrain:
            self.model_names = ['G_A', 'F1', 'D_A', 'G_B', 'F2', 'D_B']
        else:  
            self.model_names = ['G_A', 'G_B']

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                        opt.no_antialias_up, self.gpu_ids, opt)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                        opt.no_antialias_up, self.gpu_ids, opt)
        self.netF1 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)
        self.netF2 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)
            self.fake_A_pool = ImagePool(opt.pool_size) 
            self.fake_B_pool = ImagePool(opt.pool_size)  
            
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)       
            self.criterionFrequency = FrequencyLoss()
        
            self.criterionSim = torch.nn.L1Loss('sum').to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  
        if self.opt.isTrain:
            self.compute_G_loss().backward()  
            self.backward_D_A()  
            self.backward_D_B()  
            self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF1.parameters(), self.netF2.parameters()))
            self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()  
        self.backward_D_B()  
        self.optimizer_D.step()

        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  
        self.fake_A = self.netG_B(self.real_B)  

        self.recovery_A = self.netG_B(self.fake_B)

        self.identity_A = self.netG_B(self.real_A)

        self.recovery_B = self.netG_A(self.fake_A)

        self.identity_B = self.netG_A(self.real_B)


        if self.opt.nce_idt:
            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B) * self.opt.lambda_GAN

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) * self.opt.lambda_GAN

    def compute_G_loss(self):
        fakeB = self.fake_B
        fakeA = self.fake_A

        self.fake_B_red = self.fake_B[:, 0:1, :, :]
        self.fake_B_green = self.fake_B[:, 1:2, :, :]
        self.fake_B_blue = self.fake_B[:, 2:3, :, :]

        self.real_B_red = self.real_B[:, 0:1, :, :]
        self.real_B_green = self.real_B[:, 1:2, :, :]
        self.real_B_blue = self.real_B[:, 2:3, :, :]

        self.fake_A_red = self.fake_A[:, 0:1, :, :]
        self.fake_A_green = self.fake_A[:, 1:2, :, :]
        self.fake_A_blue = self.fake_A[:, 2:3, :, :]

        self.real_A_red = self.real_A[:, 0:1, :, :]
        self.real_A_green = self.real_A[:, 1:2, :, :]
        self.real_A_blue = self.real_A[:, 2:3, :, :]

        self.loss_G_L1 = (self.criterionL1(self.fake_B_red, self.real_B_red) + self.criterionL1(self.fake_B_green,self.real_B_green) + self.criterionL1(self.fake_B_blue, self.real_B_blue)) * self.opt.lambda_L1 + self.criterionL1(self.fake_B,self.real_B) * self.opt.lambda_L1 + self.criterionL1(self.recovery_A, self.real_A) * self.opt.cyc_L1 + self.criterionL1(self.identity_A,self.real_A) * self.opt.lambda_identity + (self.criterionL1(self.fake_A_red, self.real_A_red) + self.criterionL1(self.fake_A_green, self.real_A_green) + self.criterionL1(self.fake_A_blue,self.real_A_blue)) * self.opt.lambda_L1 + self.criterionL1(self.fake_A, self.real_A) * self.opt.lambda_L1 + self.criterionL1(self.recovery_B,self.real_B) * self.opt.cyc_L1 + self.criterionL1(self.identity_B, self.real_B) * self.opt.lambda_identity
        
        self.loss_G_Frequency = self.criterionFrequency(self.fake_B, self.real_B) * self.opt.lambda_feat + self.criterionFrequency(self.fake_A, self.real_A) * self.opt.lambda_feat
        
        if self.opt.lambda_GAN > 0.0:
            pred_fakeB = self.netD_A(fakeB)
            pred_fakeA = self.netD_B(fakeA)
            self.loss_G_A = self.criterionGAN(pred_fakeB, True).mean() * self.opt.lambda_GAN
            self.loss_G_B = self.criterionGAN(pred_fakeA, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_A = 0.0
            self.loss_G_B = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE1 = self.calculate_NCE_loss1(self.real_A, self.fake_B) * self.opt.lambda_NCE
            self.loss_NCE2 = self.calculate_NCE_loss2(self.real_B, self.fake_A) * self.opt.lambda_NCE
        else:
            self.loss_NCE1, self.loss_NCE_bd, self.loss_NCE2 = 0.0, 0.0, 0.0
        if self.opt.lambda_NCE > 0.0:

            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_IDT
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_IDT
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5 + (self.loss_idt_A + self.loss_idt_B) * 0.5

        else:
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5

        self.loss_G = (self.loss_G_A + self.loss_G_B) * 0.5 + loss_NCE_both

        return self.loss_G

    def calculate_NCE_loss1(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_B(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG_A(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF1(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF2(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_NCE_loss2(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG_B(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF2(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF1(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals

class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, x, target):
        b, c, h, w = x.size()
        x = x.contiguous().view(-1,h,w)
        target = target.contiguous().view(-1,h,w)
        x_fft = torch.rfft(x, signal_ndim=2, normalized=False, onesided=True)
        target_fft = torch.rfft(target, signal_ndim=2, normalized=False, onesided=True)
        
        _, h, w, f = x_fft.size()
        x_fft = x_fft.view(b,c,h,w,f)
        target_fft = target_fft.view(b,c,h,w,f)
        diff = x_fft - target_fft
        return torch.mean(torch.sum(diff**2, (1, 2, 3, 4)))
