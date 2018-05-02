from cfg import Cfg
from generator import MyGenerator
from model_architecture import Net
import numpy as np
import os
import pickle
from retrieval_eval import evaluate
import time
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.optim import Adam
from torch import from_numpy,load
from utils import activation_centered, compute_mat_pow, compute_correlation, save_checkpoint, symmetric_average

if __name__ == '__main__':
    t0 = time.time()
    cfg = Cfg()
    loss_function = MSELoss()
    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)
    if cfg.resume:
        # Loading pre-trained weights
        if os.path.isfile(cfg.resume):
            print("=> Loading checkpoint '{}'".format(cfg.resume))
            checkpoint = load(cfg.resume)
            cfg.start_epoch = checkpoint['epoch']
            cfg.batch_size_train = checkpoint['batch_size_train']
            cfg.epsilon = checkpoint['epsilon']
            cfg.lr = checkpoint['lr']
            cfg.rho = checkpoint['rho']
            cfg.n_input_sizes = checkpoint['n_input_sizes']
            cfg.layer_sizes_F = checkpoint['layer_sizes_F']
            cfg.layer_sizes_G = checkpoint['layer_sizes_G']
            cfg.activ = checkpoint['activ']
            model_F = Net(cfg.layer_sizes_F, cfg.activ)
            model_G = Net(cfg.layer_sizes_G, cfg.activ)
            model_F.load_state_dict(checkpoint['state_dict_F'])
            model_G.load_state_dict(checkpoint['state_dict_G'])
            optimizer_F = Adam(model_F.parameters(), lr=cfg.lr)
            optimizer_G = Adam(model_G.parameters(), lr=cfg.lr)
            optimizer_F.load_state_dict(checkpoint['optimizer_F'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            print("=> Loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
            print('Model F: ', model_F)
            print('Model G: ', model_G)
        else:
            print("=> No checkpoint found at '{}'".format(cfg.resume))

    else:
        # Training model
        model_F = Net(cfg.layer_sizes_F, cfg.activ)
        model_G = Net(cfg.layer_sizes_G, cfg.activ)
        optimizer_F = Adam(model_F.parameters(), lr=cfg.lr)
        optimizer_G = Adam(model_G.parameters(), lr=cfg.lr)
        print('Model F: ', model_F)
        print('Model G: ', model_G)
        # Choosing batch at random and initializing co-variances
        batch = next(MyGenerator(cfg.dataset, 'train', cfg.feats, cfg.batch_size_train,
                                 train_mode=False))
        F_init = model_F(Variable(from_numpy(batch[0]),requires_grad=False))
        G_init = model_G(Variable(from_numpy(batch[1]),requires_grad=False))
        Z_init = activation_centered(Variable(from_numpy(batch[2]),requires_grad=False),
                                           cfg.activ)
        # Initializing co-variances
        cfg.initialize_variances(F_init, G_init, Z_init)

        for epoch in range(cfg.start_epoch,cfg.training_epochs):
            print('Starting epoch number ',epoch)
            for batch in MyGenerator(cfg.dataset, 'train', cfg.feats, cfg.batch_size_train,train_mode=True):
                # Forward pass
                F_train = model_F(Variable(from_numpy(batch[0])))
                G_train = model_G(Variable(from_numpy(batch[1])))
                Z_train = activation_centered(Variable(from_numpy(batch[2]),requires_grad=False),
                                              cfg.activ)
                # Updating co-variances
                cfg.update_variances(F_train, G_train, Z_train)
                # Computing conditional variables and co-variances
                cfg.update_conditional_variables(F_train, G_train, Z_train)
                # Computing right side of the loss
                FF_Z_inv_half = compute_mat_pow(cfg.FF_Z, -0.5, cfg.epsilon)
                GG_Z_inv_half = compute_mat_pow(cfg.GG_Z, -0.5, cfg.epsilon)
                FF_Z_inv_half = symmetric_average(FF_Z_inv_half)
                GG_Z_inv_half = symmetric_average(GG_Z_inv_half)
                # Fixing right side of the loss
                F_pred = (cfg.F_Z).mm(FF_Z_inv_half).detach()
                G_pred = (cfg.G_Z).mm(GG_Z_inv_half).detach()
                # Computing loss
                loss_F = loss_function(cfg.F_Z, G_pred)
                loss_G = loss_function(cfg.G_Z, F_pred)
                # Checking for nan's
                if np.isnan(loss_F.data.numpy()) or np.isnan(loss_G.data.numpy()):
                    raise SystemExit('loss is Nan')
                # Reseting gradients, performing a backward pass, and updating the weights
                optimizer_F.zero_grad()
                loss_F.backward()
                optimizer_F.step()
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

            if epoch % cfg.display_val_step == 0:
                # Evaluating on validation set
                print('Loss F is: ',loss_F.data.numpy()[0])
                print('Loss G is: ',loss_G.data.numpy()[0])
                split = 'val'
                batch = next(MyGenerator(cfg.dataset, split, cfg.feats, cfg.batch_size_train,
                                         train_mode=False))
                F_val = model_F(Variable(from_numpy(batch[0]), volatile=True))
                G_val = model_G(Variable(from_numpy(batch[1]), volatile=True))
                Z_val = activation_centered(Variable(from_numpy(batch[2]), volatile=True),
                                            cfg.activ)
                print('== Evaluating on ' + split + ' set ==')
                print('Correlation of F and G is: ', compute_correlation(F_val, G_val).data.numpy()[0])
                F_res, G_res = evaluate(F_val, G_val, cfg)
                is_best = F_res + G_res > 0.0
                if is_best:
                    # Saving checkpoint
                    save_checkpoint({'epoch': epoch + 1, 'state_dict_F': model_F.state_dict(),
                    'state_dict_G': model_G.state_dict(), 'optimizer_F': optimizer_F.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(), 'batch_size_train': cfg.batch_size_train,
                    'epsilon': cfg.epsilon, 'lr': cfg.lr, 'rho': cfg.rho,
                    'n_input_sizes': cfg.n_input_sizes, 'layer_sizes_F': cfg.layer_sizes_F,
                    'layer_sizes_G': cfg.layer_sizes_G, 'activ': cfg.activ},
                    './outputs/dpcca_a_' + cfg.comment + cfg.dname_ + '_F_res_' + str(F_res) +
                    '_G_res_' + str(G_res) + '_split_' + split + '_dict.pth.tar')
                    # Saving validation vectors
                    pickle.dump({'F': F_val, 'G': G_val, 'Z': Z_val},
                                open('./outputs/dpcca_a_' + cfg.comment + cfg.dname_ +
                                     '_F_res_' + str(F_res) + '_G_res_' +
                                     str(G_res) + '_split_' + split + '_vectors.p', 'wb'))
    # Test #
    split = 'test'
    batch = next(MyGenerator(cfg.dataset, split, cfg.feats, cfg.batch_size_train,
                             train_mode=False))
    F_test = model_F(Variable(from_numpy(batch[0]), volatile=True))
    G_test = model_G(Variable(from_numpy(batch[1]), volatile=True))
    Z_test = activation_centered(Variable(from_numpy(batch[2]), volatile=True), cfg.activ)
    print('== Evaluating on ' + split + ' set ==')
    print('Correlation of F and G is: ', compute_correlation(F_test, G_test).data.numpy()[0])
    # Evaluating on test set
    F_res, G_res = evaluate(F_test, G_test, cfg)

    # Saving test vectors
    pickle.dump({'F': F_test, 'G': G_test, 'Z': Z_test},
                open('./outputs/dpcca_a_' + cfg.comment + cfg.dname_ +
                     '_F_res_' + str(F_res) + '_G_res_' +
                     str(G_res) + '_split_' + split + '_vectors.p', 'wb'))
    print(time.time() - t0, 'seconds wall time')




