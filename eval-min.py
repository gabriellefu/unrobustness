"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

    
from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed
from core import setup


import tensorflow as tf
import time
import scipy.io
import sys


import torch.multiprocessing as mp
mp.set_start_method('spawn')

tf.compat.v1.disable_eager_execution()

from fab_attack import FAB_linf
from fab_attack import FAB_l1
from fab_attack import FAB_l2

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Setup

parse = parser_eval()
args = parse.parse_args()

LOG_DIR = os.path.join(args.log_dir, args.desc)
with open(os.path.join(LOG_DIR, 'args.txt'), 'r') as f:
    old = json.load(f)
    old['data_dir'], old['log_dir'] = args.data_dir, args.log_dir
    args.__dict__ = dict(vars(args), **old)

DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')

if 'imagenet' in args.data:
    setup.setup_train(DATA_DIR)
    setup.setup_val(DATA_DIR)
    args.data_dir = os.environ['TMPDIR']
    DATA_DIR = os.path.join(args.data_dir, args.data)

log_path = os.path.join(LOG_DIR, 'log-aa.log')
logger = Logger(log_path)

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))



# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)



# Model

model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint



# FAB Evaluation

seed(args.seed)
sess = tf.compat.v1.InteractiveSession()
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
class HParams:
    def __init__(self):
        self.n_iter = 10
        self.n_restarts = 0
        self.eps = 0.0
        self.alpha_max = 0.0
        self.n_labels = 0
        self.targetcl = -1
        self.final_search = False
        self.bs = 256
        self.im = 100
        self.dataset = 'cifar10'

# 创建一个 HParams 对象
hps = HParams()

hps.n_iter = 10
hps.n_restarts =3
hps.bs=BATCH_SIZE
hps.n_labels=10
hps.eps=1/255


test_point = 100
t1 = time.time()

num_samples = len(test_dataloader.dataset)
adv = np.zeros((num_samples, *x_test.shape[1:]))
res = np.zeros(num_samples)

sp = 0
for i, (x_batch, y_batch) in enumerate(test_dataloader):
    if hps.p == 'linf':
        batch_res, batch_adv = FAB_linf.FABattack_linf(model, x_batch.numpy(), y_batch.numpy(), sess, hps)
    elif hps.p == 'l2':
        batch_res, batch_adv = FAB_l2.FABattack_l2(model, x_batch.numpy(), y_batch.numpy(), sess, hps)
    elif hps.p == 'l1':
        batch_res, batch_adv = FAB_l1.FABattack_l1(model, x_batch.numpy(), y_batch.numpy(), sess, hps)

    batch_size = x_batch.shape[0]
    res[sp:sp + batch_size] = batch_res
    adv[sp:sp + batch_size] = batch_adv

    sp += batch_size

t1 = time.time() - t1
print('attack performed in {:.2f} s'.format(t1))
print('misclassified points: {:d}'.format(np.sum(res == 0)))
print('success rate: {:d}/{:d}'.format(np.sum((res > 0) * (res < 1e10)), np.sum(res > 0)))
print('average perturbation size: {:.5f}'.format(np.mean(res[(res > 0) * (res < 1e10)])))

pred = sess.run(model.corr_pred, {model.x_input: adv, model.y_input: y_test[:len(test_dataloader.dataset)], model.bs: len(test_dataloader.dataset)})
print('robust accuracy: {:.2f}%'.format(np.mean(pred.astype(int)) * 100))

sess.close()

print ('Script Completed.')

