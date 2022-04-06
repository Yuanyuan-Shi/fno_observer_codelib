
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)


class GRU(nn.Module):
    def __init__(self, layer, width, x_size=512):
        super(GRU, self).__init__()

        self.num_layers = layer
        self.hidden_size = width
        self.x_size = x_size

        self.gru = nn.GRU(input_size=x_size, hidden_size=width, num_layers=layer)

        self.fc = nn.Linear(width, x_size)

    def forward(self, x, h=None):

        T_size = x.shape[0]
        batch_size = x.shape[1]

        # h, c (num_layers * num_directions, batch, hidden_size)
        if h ==None:
            h = self.init_hidden(shape=(self.num_layers, batch_size, self.hidden_size), device=x.device)

        #input (seq_len, batch, input_size)
        out, h = self.gru(x.view(T_size, batch_size, self.x_size), h)

        out = self.fc(out)

        return out, h

    def init_hidden(self, shape, device):
        return torch.zeros(shape, device=device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

# PATH_DATA = 'data/KS_L64pi_N1200_s2048_T200_t200.mat'
PATH_DATA = 'data/KS_L64pi_N1200_s512_T500_t2000.mat'

total_index = 0
total_error = np.zeros((6,500,2))

for sub in [1, ]:

    Ntrain = 1000 # training instances
    Ntest = 200 # testing instances
    T_in = 200
    T_out = 400
    T = 10
    T_iter = ((T_out - T_in)//T)
    t = T # timesteps

    ntrain = Ntrain * T_iter * T
    ntest = Ntest * T_iter * T

    # sub = 1 #subsampling rate
    s = 512 // sub

    print(Ntrain, Ntest, ntrain, ntest)
    print(T_in, T_out, T, T_iter, s)

    batch_size = 50
    learning_rate = 0.001

    epochs = 200
    step_size = 20
    gamma = 0.5
    ep_print = 10

    layer = 1
    width = 1000

    print(layer, width, batch_size, learning_rate, epochs)


    path = 'KS_gru_N'+str(ntrain)+ '_s' + str(s) +'_ep' + str(epochs) + '_l' + str(layer) + '_w' + str(width)
    path_model = 'model/'+path
    path_pred = 'pred/'+path+'.mat'
    path_error = 'results/'+path+'train.txt'


    dataloader = MatReader(PATH_DATA)
    x_train = dataloader.read_field('u')[:Ntrain, T_in-1:T_out-1, ::sub].reshape(Ntrain,T_iter,t,s)
    y_train = dataloader.read_field('u')[:Ntrain, T_in:T_out, ::sub].reshape(Ntrain,T_iter,t,s)
    print(x_train.shape)
    print(y_train.shape)
    x_test = dataloader.read_field('u')[-Ntest:, T_in-1:T_out-1, ::sub].reshape(Ntest,T_iter,t,s)
    y_test = dataloader.read_field('u')[-Ntest:, T_in:T_out, ::sub].reshape(Ntest,T_iter,t,s)



    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = GRU(layer=layer, width=width, x_size=s).cuda()
    # model = torch.load('model/KS_gru_N1500000_s512_ep50_l1_w1000')

    print(model.count_params())

    myloss = LpLoss(size_average=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    error = np.zeros((500, 2))
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_traj = np.zeros(T_iter, )

        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            h = None

            for i in range(0, T_iter):

                # xx,yy: (batch, T_iter, t, s)
                x = xx[:, i, :, :].permute(1,0,2) # (t, batch, s)
                y = yy[:, i, :, :].permute(1,0,2) # (t, batch, s)

                im, h = model(x, h)

                h = h.detach()

                optimizer.zero_grad()
                loss = myloss(im.reshape(-1, s), y.reshape(-1, s))
                loss.backward()
                optimizer.step()

                train_traj[i] += loss.item()
                train_l2 += loss.item()

        if ep % ep_print == ep_print-1:
            test_l2 = 0
            test_traj = np.zeros(T_iter, )

            model.eval()
            with torch.no_grad():
                for xx, yy in test_loader:
                    xx = xx.to(device)
                    yy = yy.to(device)
                    h = None
                    for i in range(0, T_iter):
                        # xx,yy: (batch, T_iter, t, s)
                        x = xx[:, i, :, :].permute(1, 0, 2)  # (t, batch, s)
                        y = yy[:, i, :, :].permute(1, 0, 2)  # (t, batch, s)

                        im, h = model(x, h)

                        loss = myloss(im.reshape(-1, s), y.reshape(-1, s))
                        test_traj[i] += loss.item()
                        test_l2 += loss.item()

            t2 = default_timer()

            train_l2 = train_l2 / (T_iter*Ntrain*T)
            train_traj = train_traj / (Ntrain*T)
            test_l2 = test_l2 / (T_iter*Ntest*T)
            test_traj = test_traj / (Ntest*T)

            print(ep, t2 - t1, train_l2, test_l2)
            # print(ep, t2 - t1, train_l2, train_traj, test_l2, test_traj)
            error[ep] = [train_l2, test_l2]

        scheduler.step()
    torch.save(model, path_model )


    # test
    T_in = 1000
    T_out = 2000
    T_warmup = 1500
    T_iter = (T_out-T_in)
    print(T_out, T_warmup)

    dataloader = MatReader(PATH_DATA)
    x_test = dataloader.read_field('u')[0, (T_in-1), ::sub].reshape(1, s)
    y_test = dataloader.read_field('u')[0, T_in:T_out, ::sub].reshape(T_iter, s)

    print(x_test.shape)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_test), batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        pred = torch.zeros(T_iter, s)
        errors = torch.zeros(T_iter, )
        index = 0
        out = x_test.cuda()
        h = None
        #no warm up
        for y, in test_loader:
            x_in = out.view(1, 1, s)
            y = y.cuda()

            out, h = model(x_in, h)
            pred[index] = out.reshape(1,s)

            l2 = myloss(out.view(1, -1), y.view(1, -1)).item()
            errors[index] = l2
            # if index % ep_print == ep_print-1:
            #     print(index, l2)
            index = index + 1

        #warm up
        pred2 = torch.zeros(T_iter, s)
        errors2 = torch.zeros(T_iter, )
        index = 0
        out = x_test.cuda()
        h = None
        for y, in test_loader:
            x_in = out.view(1, 1, s)
            y = y.cuda()

            out, h = model(x_in, h)
            pred2[index] = out.reshape(1,s)

            l2 = myloss(out.view(1, -1), y.view(1, -1)).item()
            errors2[index] = l2
            # if index % ep_print == ep_print-1:
            #     print(index, l2)
            index = index + 1

            if index+T_in < T_warmup:
                out = y

    scipy.io.savemat(path_pred, mdict={'pred': pred.cpu().numpy(), 'pred2': pred2.cpu().numpy(), 'y': y_test.cpu().numpy(),})

    total_error[total_index] = error
    total_index = total_index + 1

scipy.io.savemat(path_error, mdict={'error': total_error})
