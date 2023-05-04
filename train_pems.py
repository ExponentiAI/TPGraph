import time
import datetime
from data.utils_pems import log_string
# from model.model_ import *
from data.utils_pems import load_data, count_parameters
# import math
from TPGraph import TPGraph
import torch
from data.generate_adj_pems import read_adj
import pandas as pd

def train(model, args, log, loss_criterion, optimizer, scheduler):

    (train_loader, val_loader, test_loader,
     SE, mean, std, ifo) = load_data(args)

    wait = 0
    val_loss_min = float('inf')
    test_loss_min = float('inf')
    best_model_wts = None
    train_total_loss = []
    val_total_loss = []
    test_total_loss = []

    for epoch in range(args.max_epoch):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        start_train = time.time()
        model.train()
        train_loss = 0
        num_train = 0
        for ind, data in enumerate(train_loader):
            xc, xd, xw, te, y = data  # B T N -> x need: B C N T
            xc, xd, xw = xc.unsqueeze(1).permute(0,1,3,2), xd.unsqueeze(1).permute(0,1,3,2), xw.unsqueeze(1).permute(0,1,3,2)
            optimizer.zero_grad()
            pred = model(xc, xd, xw, te)  # 32 12 325 B T N
            pred = pred * std + mean
            loss_batch = loss_criterion(pred, y)
            num_train += xc.shape[0]
            train_loss += float(loss_batch) * xc.shape[0]
            loss_batch.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (ind + 1) % 20 == 0:
                print(f'Training batch: {ind + 1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}')
            del xc, xd, xw, y, loss_batch
        train_loss /= num_train
        train_total_loss.append(train_loss)
        end_train = time.time()

        # val start
        start_val = time.time()
        val_loss = 0
        num_val = 0
        model.eval()
        with torch.no_grad():
            for ind, data in enumerate(val_loader):
                xc, xd, xw, te, y = data  # B T N -> x need: B C N T； te：B 2T 2
                xc, xd, xw = xc.unsqueeze(1).permute(0,1,3,2), xd.unsqueeze(1).permute(0,1,3,2), xw.unsqueeze(1).permute(0,1,3,2)
                optimizer.zero_grad()
                pred = model(xc, xd, xw, te)  # 32 12 325 B T N
                pred = pred * std + mean
                loss_batch = loss_criterion(pred, y)
                val_loss += loss_batch * xc.shape[0]
                num_val += xc.shape[0]
                del xc, xd, xw, y, loss_batch
        val_loss /= num_val
        val_total_loss.append(val_loss)
        end_val = time.time()
        log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             args.max_epoch, end_train - start_train, end_val - start_val))
        log_string(
            log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
        if val_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {args.model_file}')
            wait = 0
            val_loss_min = val_loss
            best_model_wts = model.state_dict()
            torch.save(model, 'ST_PEM_ez{0}_layer{1}_epoch{2}'.format(args.embed_size, args.num_layers, epoch))  #
        else:
            wait += 1

        scheduler.step()

    model.load_state_dict(best_model_wts)
    torch.save(model, args.model_file)
    log_string(log, f'Training and validation are completed, and model has been stored as {args.model_file}')
    return train_total_loss, val_total_loss


