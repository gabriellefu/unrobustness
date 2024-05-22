import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import create_attack
from core.metrics import accuracy
from .context import ctx_noparamgrad_and_eval

from torch.autograd import Variable




def wrong_loss(model, x, y, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=10, 
                attack='linf-pgd', hr_model=None):
    """
    AT + Helper-based adversarial training.
    """
  
    criterion_ce = nn.CrossEntropyLoss()
    model.eval()
  
    attack = create_attack(model, criterion_ce, attack, epsilon, perturb_steps, step_size)
    with ctx_noparamgrad_and_eval(model):
        x_adv, _ = attack.perturb(x, y)
        
    model.train()
    
    with ctx_noparamgrad_and_eval(hr_model):
        out_wronglabel = hr_model(x_adv).argmax(dim=1)#generate the label of the adversaial example
  
    optimizer.zero_grad()
  
    out_clean, out_adv = model(x), model(x_adv)
    loss_clean = F.cross_entropy(out_clean, y, reduction='mean')
    loss_adv = criterion_ce(out_adv, out_wronglabel)
    loss = loss_clean + loss_adv
     
    batch_metrics = {'loss': loss.item()}
    batch_metrics.update({'adversarial_acc': accuracy(y, out_adv.detach()), 'wrong_label_acc': accuracy(out_wronglabel, out_adv.detach())}) 
    batch_metrics.update({'clean_acc': accuracy(y, out_clean.detach())})
  
    return loss, batch_metrics