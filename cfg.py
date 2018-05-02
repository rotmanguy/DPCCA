import h5py
from torch import add, t
from utils import compute_mat_pow, symmetric_average

class Cfg(object):
    def __init__(self):
        # Learning Parameters
        self.training_epochs = 100 # Number of training epochs
        self.batch_size_train = 128 # Batch size for the training set
        self.start_epoch = 0 # Initial epoch number
        self.display_val_step = 5 # Number of epochs to run until evaluating on the validation set
        self.epsilon = 1e-6 # Constant for stability
        self.lr = 1e-4  # Learning rate
        self.rho = 0.75  # Weight for the covariance estimation
        self.output_folder = './outputs/'
        self.resume = '' # Resume model weights, default None, please specify path to load pre-trained model from (tar file)

        # Network Parameters
        self.n_input_sizes = [300, 300, 300] # Input dimensions
        self.n_views = len(self.n_input_sizes) # Number of views
        self.layer_sizes_F = [self.n_input_sizes[0], 300, 300, 300, 128]
        self.layer_sizes_G = [self.n_input_sizes[1], 300, 300, 300, 128]
        self.layer_sizes_Z = [self.n_input_sizes[2], 300, 300, 300, 128] # For dpcca_b
        self.activ = 'tanh' # Activation function
        # To change number of hidden layers one should modify also the model_architecture.py accordingly

        # Dataset parameters
        self.feats = ['eng', 'ru', 'vis'] # Features # feats[1] = 'ger' / 'it' / 'ru'
        self.lng_dict = {'eng': 'english', 'ger': 'german', 'it': 'italian', 'ru': 'russian'}
        self.languages = [self.lng_dict[self.feats[0]], self.lng_dict[self.feats[1]]]
        self.folder = self.feats[0] + '_' + self.feats[1]
        self.dname_ = 'wiw_' + self.folder + '_img' # Dataset Name
        self.dname =  self.dname_ + '_pca_' + str(self.n_input_sizes[2]) + '_wv_splitted'
        self.dataset = h5py.File('./data/wiw_data/' + self.folder + '/' + self.dname + '.h5', 'r')
        self.comment = ''
        self.train_size = self.count_split_size('train')
        self.val_size = self.count_split_size('val')
        self.test_size = self.count_split_size('test')

    def count_split_size(self,split):
        dataset_split = self.dataset[split]
        split_size = 0
        for _ in dataset_split:
            split_size += 1
        return split_size

    def initialize_variances(self, F, G, Z):
        # Initializing co-variances
        batch_size = F.size()[0]
        div = float(self.train_size) / batch_size
        self.FF = (t(F)).mm(F).mul(div)
        self.GG = (t(G)).mm(G).mul(div)
        self.ZZ = (t(Z)).mm(Z).mul(div)
        self.ZF = (t(Z)).mm(F).mul(div)
        self.ZG = (t(Z)).mm(G).mul(div)
        self.FF = symmetric_average(self.FF) # Ensuring matrix is symmetric
        self.GG = symmetric_average(self.GG) # Ensuring matrix is symmetric
        self.ZZ = symmetric_average(self.ZZ) # Ensuring matrix is symmetric

    def initialize_variances_no_Z(self, F, G):
        # Initializing co-variances
        batch_size = F.size()[0]
        div = float(self.train_size) / batch_size
        self.FF = (t(F)).mm(F).mul(div)
        self.GG = (t(G)).mm(G).mul(div)
        self.FF = symmetric_average(self.FF) # Ensuring matrix is symmetric
        self.GG = symmetric_average(self.GG) # Ensuring matrix is symmetric

    def update_variances(self, F, G, Z):
        # Updating co-variances
        batch_size = F.size()[0]
        rho = self.rho
        div = float(self.train_size) / batch_size
        one_minus_rho_times_div = (1 - rho) * div
        self.FF = add(((self.FF).mul(rho)).detach(), t(F).mm(F).mul(one_minus_rho_times_div))
        self.GG = add(((self.GG).mul(rho)).detach(), t(G).mm(G).mul(one_minus_rho_times_div))
        self.ZZ = add(((self.ZZ).mul(rho)).detach(), t(Z).mm(Z).mul(one_minus_rho_times_div))
        self.ZF = add(((self.ZF).mul(rho)).detach(), t(Z).mm(F).mul(one_minus_rho_times_div))
        self.ZG = add(((self.ZG).mul(rho)).detach(), t(Z).mm(G).mul(one_minus_rho_times_div))
        self.FF = symmetric_average(self.FF) # Ensuring matrix is symmetric
        self.GG = symmetric_average(self.GG) # Ensuring matrix is symmetric
        self.ZZ = symmetric_average(self.ZZ) # Ensuring matrix is symmetric

    def update_variances_no_Z(self, F, G):
        # Updating co-variances
        batch_size = F.size()[0]
        rho = self.rho
        div = float(self.train_size) / batch_size
        one_minus_rho_times_div = (1 - rho) * div
        self.FF = add(((self.FF).mul(rho)).detach(), t(F).mm(F).mul(one_minus_rho_times_div))
        self.GG = add(((self.GG).mul(rho)).detach(), t(G).mm(G).mul(one_minus_rho_times_div))
        self.FF = symmetric_average(self.FF) # Ensuring matrix is symmetric
        self.GG = symmetric_average(self.GG) # Ensuring matrix is symmetric


    def update_conditional_variables(self, F, G, Z):
        # Calculating conditional variables; F_Z, G_Z, FF_Z, GG_Z
        self.ZZ_inverse = compute_mat_pow(self.ZZ, -1, self.epsilon) # Computing inverse of Sigma_ZZ
        self.ZZ_inverse = symmetric_average(self.ZZ_inverse) # Ensuring matrix is symmetric
        self.ZZ_inverse_mul_ZF = (self.ZZ_inverse).mm(self.ZF)
        self.ZZ_inverse_mul_ZG = (self.ZZ_inverse).mm(self.ZG)
        self.mu_F_Z = Z.mm(self.ZZ_inverse_mul_ZF)
        self.mu_G_Z = Z.mm(self.ZZ_inverse_mul_ZG)
        self.F_Z = F.sub(self.mu_F_Z) # F given Z
        self.G_Z = G.sub(self.mu_G_Z) # G given Z
        self.mu_F_Z_mu_F_Z = (t(self.ZF)).mm(self.ZZ_inverse_mul_ZF)
        self.mu_G_Z_mu_G_Z = (t(self.ZG)).mm(self.ZZ_inverse_mul_ZG)
        self.mu_F_z_mu_F_Z = symmetric_average(self.mu_F_Z_mu_F_Z)
        self.mu_G_Z_mu_G_Z = symmetric_average(self.mu_G_Z_mu_G_Z)
        self.FF_Z = (self.FF).sub(self.mu_F_Z_mu_F_Z) # Sigma_FF given Z
        self.GG_Z = (self.GG).sub(self.mu_G_Z_mu_G_Z) # Sigma_GG given Z
        self.FF_Z = symmetric_average(self.FF_Z) # Ensuring matrix is symmetric
        self.GG_Z = symmetric_average(self.GG_Z) # Ensuring matrix is symmetric