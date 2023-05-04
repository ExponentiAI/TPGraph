import time
from data.utils_pems import log_string, plot_train_val_loss
from data.utils_pems import load_data, count_parameters
from TPGraph import TPGraph
from data.generate_adj_pems import read_adj
import pandas as pd
from train_pems import train
from test_pems import test
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    from args_parameter_pems import *

    # log
    log = open(args.log_file+'_epoch{0}'.format(args.max_epoch), 'w')
    log_string(log, str(args)[10: -1])
    T = 24 * 60 // args.time_slot
    log_string(log, 'loading data...')
    (train_loader, val_loader, test_loader,
     SE, mean, std, ifo) = load_data(args)

    trainshape, valshape, testshape = ifo['trainshape'], ifo['valshape'], ifo['testshape']
    log_string(log, f'trainX: {trainshape[0:3]}\t\t trainY: {trainshape[-1]}')
    log_string(log, f'valX:   {valshape[0:3]}\t\tvalY:   {valshape[-1]}')
    log_string(log, f'testX:   {testshape[0:3]}\t\ttestY:   {trainshape[-1]}')
    log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
    log_string(log, 'data loaded!')
    del trainshape, valshape, testshape, mean, std, ifo
    log_string(log, 'compiling model...')

    # model initiation
    adj_w, adj_r = read_adj('data/Adj(PeMS).txt')  # adj_w ranges from 0 to 1; -r is 0 or 1
    adj_w, adj_r = adj_w.float(), adj_r.float()
    df = pd.read_hdf('data/pems_correlation_roads.h5')
    data_val = torch.from_numpy(df.values)  # 325*325
    corr = torch.Tensor(data_val).float()

    in_channels = 1  # Channels of input
    embed_size = args.embed_size  # Dimension of hidden embedding features
    time_num = T  # 288
    num_layers = args.num_layers  # Number of ST Block
    T_dim = [args.num_his, args.num_day, args.num_week]  # Input length
    output_T_dim = args.num_pred  # Output Expected length
    heads = args.heads  # Number of Heads in MultiHeadAttention
    cheb_K = 2
    forward_expansion = 4  # Dimension of Feed Forward Network
    dropout = args.drop  # regression tasks do not use dropout
    extra_feature = torch.Tensor(120, 2)  # optional ifo
    model = TPGraph(
        adj_r,
        corr,
        extra_feature,
        in_channels,
        embed_size,
        time_num,
        num_layers,
        T_dim,
        output_T_dim,
        heads,
        cheb_K,
        forward_expansion,
        args.DEVICE,
        dropout=dropout)

    model.to(args.DEVICE)

    # loss and optimizer
    loss_criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.5)
    parameters = count_parameters(model)
    log_string(log, 'trainable parameters: {:,}'.format(parameters))

    start = time.time()

    # training
    loss_train, loss_val = train(model, args, log, loss_criterion, optimizer, scheduler)
    plot_train_val_loss(loss_train, loss_val, 'train_val_loss_pre.png')

    # testing
    trainPred, valPred, testPred, trainY, valY, testY = test(args, model, log)
    end = time.time()
    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()
    trainPred_ = trainPred.numpy().reshape(-1, 325)
    trainY_ = trainY.numpy().reshape(-1, 325)
    valPred_ = valPred.numpy().reshape(-1, 325)
    valY_ = valY.numpy().reshape(-1, 325)
    testPred_ = testPred.numpy().reshape(-1, 325)
    testY_ = testY.numpy().reshape(-1, 325)

    # Save training, validation and testing datas to disk
    l = [trainPred_, trainY_, valPred_, valY_, testPred_, testY_]
    name = ['trainPred', 'trainY', 'valPred', 'valY', 'testPred', 'testY']
    for i, data in enumerate(l):
        np.savetxt('./figure_pems/' + name[i] + '{0}'.format(args.num_pred) + '.txt', data, fmt='%s')
        print('{0} saved'.format(name[i]))


    # Plot the test prediction vs target（optional)

    # plt.figure_pems(figsize=(10, 280))
    # for k in range(10):
    #     plt.subplot(10, 1, k + 1)
    #     for j in range(len(testPred)):
    #         c, d = [], []
    #         for i in range(12):
    #             c.append(testPred[j, i, k])
    #             d.append(testY[j, i, k])
    #         plt.plot(range(1 + j, 12 + 1 + j), c, c='b')
    #         plt.plot(range(1 + j, 12 + 1 + j), d, c='r')
    #     print('{0} plot success'.format(+1))
    # plt.title('Test prediction vs Target')
    # plt.savefig('./figure_pems/test_results.png')