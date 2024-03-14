import torch
import torch.jit
import torch.nn.functional as F
import itertools
from torchvision import transforms as T
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


# inserted by houxingzhong on 20210105
def create_window(window_size: int, sigma: float, channel: int):
    '''
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    '''
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


def _gaussian_filter(x, window_1d, use_padding: bool):
    '''
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    '''
    C = x.shape[1]
    padding = 0
    if use_padding:
        window_size = window_1d.shape[3]
        padding = window_size // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out


def ssim(X, Y, window, data_range: float, use_padding: bool=False):
    '''
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return:
    '''

    K1 = 0.01
    K2 = 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    C3 = C2 / 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    # Inserted by houxingzhong on 20210106
    # When sqrt is used, the gradient will be nan if the sqrt parameter is zero. So add δ to avoid gradient explosion.
    sigma1 = torch.sqrt(sigma1_sq + 0.00001)
    sigma2 = torch.sqrt(sigma2_sq + 0.00001)

    s_map = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    s_map = s_map.clamp_min(0.)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    # Fixed the issue that the negative value of cs_map caused ms_ssim to output Nan.
    cs_map = cs_map.clamp_min(0.)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map / s_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW
    # cs = cs_map.mean(dim=(1, 2, 3))
    s = s_map.mean(dim=(1, 2, 3))

    return ssim_val, s


class SSIM(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False):
        '''
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels (default: 3)
        :param use_padding: padding image before conv
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        self.data_range = data_range
        self.use_padding = use_padding


    def forward(self, X, Y):
        r = ssim(X, Y, window=self.window, data_range=self.data_range, use_padding=self.use_padding)
        return 1. - r[0]


class Stct(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False):
        '''
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels (default: 3)
        :param use_padding: padding image before conv
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        self.data_range = data_range
        self.use_padding = use_padding


    def forward(self, X, Y):
        r = ssim(X, Y, window=self.window, data_range=self.data_range, use_padding=self.use_padding)
        return 1. - r[1]


# original cyclegan
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.opt.with_mask:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'ssim_B']
        else:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'ssim_A', 'ssim_B']
        if self.opt.model_M:
            self.loss_names += ['D_M', 'idt_M']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            # visual_names_A.append('idt_B')
            # visual_names_B.append('idt_A')
        # self.visual_names = visual_names_A + visual_names_B # combine visualizations for A and B
        # feature_real_A, feature_fake_B, feature_rec_A, feature_real_B, feature_fake_A, feature_rec_B
        # fake_A, rec_B, fake_B, rec_A, back_A, back_B

        if self.isTrain:
            if self.opt.model_M:
                self.visual_names = ['input_A', 'real_A', 'fake_A', 'rec_A', 'input_B', 'real_B', 'fake_B', 'rec_B']
            elif self.opt.with_mask:
                self.visual_names = ['input_A', 'fake_A', 'rec_A', 'input_B', 'fake_B', 'rec_B', 'mask']
            else:
                self.visual_names = ['real_A', 'fake_A', 'rec_A', 'real_B', 'fake_B', 'rec_B']
        else:
            if self.opt.model_M:
                self.visual_names = ['input_A', 'real_A', 'fake_B']
            else:
                self.visual_names = ['fake_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            if self.opt.model_M:
                self.model_names += ['M', 'D_M']
            # self.model_names = ['d_A', 'r_A', 'u_A', 'd_B', 'r_B', 'u_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
            if self.opt.model_M:
                self.model_names += ['M']
            # self.model_names = ['d_A', 'r_A', 'u_A', 'd_B', 'r_B', 'u_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.

        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.G_norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.G_norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.opt.model_M:
            self.netM = networks.define_M(opt.output_nc, opt.G_norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # print(self.netG_A)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.D_norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.D_norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if self.opt.classify:
                self.netC_A = torch.load(self.opt.he_classifier)
                self.netC_B = torch.load(self.opt.ck_classifier)
                self.netC_A.to('cuda')
                self.netC_B.to('cuda')
                self.netC_A.eval()
                self.netC_B.eval()
            if self.opt.model_M:
                self.netD_M = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.D_norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netC_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netC_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            if not self.opt.serial_batches:
                self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
                self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # self.real_M_pool = ImagePool(opt.pool_size)
            self.fake_M_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionBack = torch.nn.L1Loss()
            self.criterionFeature = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionSSIM = SSIM().cuda()
            self.criterionStct = Stct().cuda()
            self.criterionClass = torch.nn.L1Loss()
            # self.criterionCls = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if self.opt.model_M:
                self.optimizer_M = torch.optim.Adam(itertools.chain(self.netM.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_M = torch.optim.Adam(itertools.chain(self.netD_M.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_M)
                self.optimizers.append(self.optimizer_D_M)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.input_A = input['A' if AtoB else 'B'].to(self.device)
        self.input_B = input['B' if AtoB else 'A'].to(self.device)
        if self.opt.with_mask:
            self.A_mask = input['A_mask'].to(self.device)
            self.mask = input['mask'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.M_T = self.image_paths[0].split('/')[-1].split('_')[0] == '1M03'

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.model_M:
            self.real_A = self.netM(self.input_A)
        elif self.opt.with_mask:
            self.real_A = self.A_mask
        else:
            self.real_A = self.input_A
        self.real_B = self.input_B
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        if self.opt.serial_batches:
            fake_B = self.fake_B
        else:
            fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        if self.opt.serial_batches:
            fake_A = self.fake_A
        else:
            fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    # Inserted by houxingzhong on 20210510
    def backward_D_M(self):
        fake_M = self.fake_M_pool.query(self.real_A)
        pred_fake = self.netD_M(fake_M.detach())
        self.loss_D_M_fake = self.criterionGAN(pred_fake, False)
        if self.M_T:
            pred_real = self.netD_M(self.input_A)
            self.loss_D_M_real = self.criterionGAN(pred_real, True)
        else:
            self.loss_D_M_real = 0

        self.loss_D_M = (self.loss_D_M_real + self.loss_D_M_fake) * 0.5

        self.loss_D_M.backward()

    # def backward_C_A(self):
    #     """Calculate GAN loss for discriminator D_A"""
    #     fake_B = self.fake_B_pool.query(self.fake_B)
    #     self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    # def backward_C_B(self):
    #     """Calculate GAN loss for discriminator D_B"""
    #     fake_A = self.fake_A_pool.query(self.fake_A)
    #     self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_ssim = self.opt.lambda_ssim
        lambda_stct = self.opt.lambda_stct
        lambda_classify = self.opt.lambda_classify
        # Identity loss
        if self.opt.idt_loss:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A.detach())
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A.detach()) * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss ||G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A.detach()) * lambda_A
        # Backward cycle loss ||G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        if self.opt.netG == 'resnet_german':
            self.loss_embedding_A = self.criterionFeature(self.feature_A1, self.feature_A2) * lambda_B
            self.loss_embedding_B = self.criterionFeature(self.feature_B1, self.feature_B2) * lambda_B
        else:
            self.loss_embedding_A = 0
            self.loss_embedding_B = 0

        # Inserted by houxingzhong on 20210415
        # self.loss_feature_cycle_A = self.criterionFeature(self.feature_A, self.feature_AA) * lambda_B
        # self.loss_feature_cycle_B = self.criterionFeature(self.feature_B, self.feature_BB) * lambda_B
        # self.loss_feature_cycle = self.loss_feature_cycle_A + self.loss_feature_cycle_B

        # Inserted by houxingzhong on 20210105
        # Calculate ssim on aligned dataset
        if self.opt.ssim_loss:
            # SSIM loss: ssim(G_A(A), A)
            if not self.opt.with_mask:
                self.loss_ssim_A = self.criterionSSIM(self.fake_A, self.real_A) * lambda_ssim
            # SSIM loss: ssim(G_B(B), B)
            self.loss_ssim_B = self.criterionSSIM(self.fake_B, self.real_B) * lambda_ssim
        else:
            self.loss_ssim_A = 0
            self.loss_ssim_B = 0

        # Inserted by houxingzhong on 20210107
        # Structural preserving
        if self.opt.stct_loss:
            # Stct loss: stct(G_A(A), A)
            # Stct loss: stct(G_B(B), B)
            self.loss_stct_A = self.criterionStct(self.fake_B, self.real_A) * lambda_stct
            self.loss_stct_B = self.criterionStct(self.fake_A, self.real_B) * lambda_stct
        else:
            self.loss_stct_A = 0
            self.loss_stct_B = 0

        # Inserted by houxingzhong on 20210601
        # Classifier loss
        if self.opt.classify:
            transforms = T.Compose([
                    T.Resize(224)
                ])
            self.loss_classify_A = self.criterionClass(self.netC_A(transforms(self.real_A)), self.netC_B(transforms(self.fake_B))) * lambda_classify
            self.loss_classify_B = self.criterionClass(self.netC_A(transforms(self.fake_A)), self.netC_B(transforms(self.real_B))) * lambda_classify
        else:
            self.loss_classify_A = 0
            self.loss_classify_B = 0

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B \
                    + self.loss_idt_A + self.loss_idt_B + self.loss_classify_A + self.loss_classify_B + self.loss_embedding_A + self.loss_embedding_B
        self.loss_G.backward()

    def backward_M(self):
        if self.M_T:
            self.loss_idt_M = self.criterionIdt(self.input_A, self.real_A)
        else:
            self.loss_idt_M = 0

        self.loss_M_T = self.criterionGAN(self.netD_M(self.real_A), True)

        self.loss_M = self.loss_idt_M + self.loss_M_T

        self.loss_M.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        if self.opt.model_M:
            self.forward()      # compute fake images, reconstruction images and features
            self.set_requires_grad([self.netD_M], False)
            self.optimizer_M.zero_grad()
            self.backward_M()
            self.optimizer_M.step()

            self.set_requires_grad([self.netD_M], True)
            self.optimizer_D_M.zero_grad()
            self.backward_D_M()
            self.optimizer_D_M.step()

        if self.opt.classify:
            self.set_requires_grad([self.netC_A, self.netC_B], False)
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
