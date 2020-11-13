# coding=utf-8
"""
Python module for implementing inner maximizers for robust adversarial training
(Table I in the paper)
"""
import torch
from torch.autograd import Variable
from utils.utils import or_float_tensors, xor_float_tensors, clip_tensor
import numpy as np
import random
import math

# helper function
def round_x(x, alpha=0.5):
    """
    rounds x by thresholding it according to alpha which can be a scalar or vector
    :param x:
    :param alpha: threshold parameter
    :return: a float tensor of 0s and 1s.
    """
    #return torch.lt(x, thresholds.cuda()).float()
    return (x > alpha).float()

def get_x0(x, is_sample=False):
    """
    Helper function to randomly initialize the the inner maximizer algos
    randomize such that the functionality is preserved.
    Functionality is preserved by maintaining the features present in x
    :param x: training sample
    :param is_sample: flag to sample randomly from feasible area or return just x
    :return: randomly sampled feasible version of x
    """
    if is_sample:
        rand_x = round_x(torch.rand(x.size()))
        if x.is_cuda:
            rand_x = rand_x.cuda()
        return or_float_tensors(x, rand_x)
    else:
        return x

def my_loss(x, x_next, target):
    loss1 = - x * (1 - x_next)
    loss2 = - torch.log(1 + (1/((10*(0.5**11))-(0.5**10))*(((x_next-0.5)**10)-(((0.5)**9)*10*((x_next-0.5)**2))+(10*(0.5**11))-(0.5**10)))) 
    return torch.mean(loss1) + torch.mean(loss2)

def newloss(x,
            y,
            model,
            loss_fct,
            k=25,
            epsilon=0.02,
            alpha=0.5,
            is_report_loss_diff=False,
            use_sample=False):

    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)
    x_var = Variable(x, requires_grad=False)

    # initialize starting point
    x_next = get_x0(x, use_sample)

    # compute natural loss
    loss_natural = loss_fct(model(x_var), y).data
    
    # multi-step
    factor = 0.01
    for n in range(k):
        # forward pass
        xn_var = Variable(x_next, requires_grad=True)
        y_model = model(xn_var)
        loss = loss_fct(y_model, y)
        factor = factor * 1.1
        factor = min(factor,10)
        loss += my_loss(x_var, xn_var, y) * factor

        # compute gradient
        grad_vars = torch.autograd.grad(loss.mean(), xn_var)[0].data
        grad_vars -= grad_vars * x

        # find the next sample
        x_next = x_next + epsilon * torch.sign(grad_vars)
        #x_next = x_next + 1000.0 * grad_vars

        # projection
        #x_next = clip_tensor(x_next)
        loss_adv = loss_fct(model(Variable(x_next)), y).data
        #print("Natural loss (%.4f) vs Adversarial loss (%.4f), With my_loss: (%.4f)" %
        #    (loss_natural.mean(), loss_adv.mean(), loss.mean()))
        #print(factor)

    # rounding step
    x_next = round_x(x_next, alpha=alpha)

    # feasible projection
    x_next = or_float_tensors(x_next, x)

    # compute adversarial loss
    loss_adv = loss_fct(model(Variable(x_next)), y).data

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
          (loss_natural.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))

    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next

def matching(x,
            y,
            model,
            loss_fct,
            k=25,
            epsilon=0.02,
            alpha=0.5,
            is_report_loss_diff=False,
            use_sample=False,
            mal_index = 0,
            dataset=None):

    mal_nrs = range(mal_index, mal_index + len(x))
    
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)

    # compute natural loss
    y_model = model(Variable(x))
    #print(y_model)
    loss_natural = loss_fct(y_model, y).data

    # initialize starting point
    x_next = x.clone()
    x_best = topk(x,y.data,model,loss_fct).cuda()
    y_model = model(Variable(x_best))
    best_loss = loss_fct(y_model, y)
     
    if dataset is None or "benign" not in dataset:
        return topk(x,y.data,model,loss_fct,k,epsilon,alpha,is_report_loss_diff,use_sample,x_next)

    for i in range(len(x)):
      for m in range(len(mal_nrs)):
        mal = mal_nrs[m]

        #print(m)

        #print(torch.sum(torch.eq(dataset["malicious"].dataset[mal][0], x[m])))

        if mal in matching_dict.keys():
            benign_matches = matching_dict[mal]
            ind = random.choice(benign_matches)
            x_next[m,:] = dataset["benign"].dataset[ind][0]
            '''#print(best_loss)
            best_row = x[m].clone()
            for ix in benign_matches:
                x_next[m,:] = dataset["benign"].dataset[ix][0]
                loss_adv = loss_fct(model(Variable(x_next[m:m+1,:])),y[m:m+1])
                #print("loss", y_ix[m][0])
                #print("\nloss", best_loss, loss_adv)
                if loss_adv.data.mean() > best_loss.data.mean():
                    best_loss = loss_adv
                    best_row = x_next[m,:].clone() 
                    #print(model(Variable(x_next[m:m+1,:])))
                    #print("\nloss", best_loss, loss_adv)
            #row = clip_tensor(row)
            #print("match:", torch.sum(torch.eq(or_float_tensors(x[m], row), row)))
            #if torch.sum(torch.eq(or_float_tensors(x[m], row), row)) != 22761:
            #    print(mal, ind)
            #    print(map_mal[mal], map_ben[ind])
            x_next[m,:] = best_row'''


      x_next = topk(x,y.data,model,loss_fct,k,epsilon,alpha,False,use_sample,x_next)
      y_model = model(Variable(x_next))
      loss_adv = loss_fct(y_model, y)
      for r in range(len(x)):
        if loss_adv.data[r] > best_loss.data[r]:
          x_best[r] = x_next[r].clone()
          best_loss.data[r] = loss_adv.data[r]
        #print(best_loss.data.mean())
      
      x_next = x_best.clone()

    # compute adversarial loss
    y_model = model(Variable(x_best))
    #print(y_model)
    loss_adv = loss_fct(y_model, y).data
    
    #if is_report_loss_diff:
    #print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
    #        (loss_natural.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))

    # projection
    #x_next = clip_tensor(x_next)
    #x_next = or_float_tensors(x_next, x)
    
    # compute adversarial loss
    #y_model = model(Variable(x_next))
    #loss_adv = loss_fct(y_model, y).data
    
    x_next = topk(x,y.data,model,loss_fct,k,epsilon,alpha,False,use_sample,x_best)
    
    # compute adversarial loss
    y_model = model(Variable(x_next))
    loss_adv = loss_fct(y_model, y).data
   
    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
            (loss_natural.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))
    
    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next


def topk(x,
            y,
            model,
            loss_fct,
            k=25,
            epsilon=0.02,
            alpha=0.5,
            is_report_loss_diff=False,
            use_sample=False,
            start_x = None):
    
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)

    # compute natural loss
    y_model = model(Variable(x))
    loss_natural = loss_fct(y_model, y).data

    # initialize starting point
    x_next = get_x0(x, use_sample).clone()
    if start_x is not None:
        x_next = start_x.clone()

    x_best = x_next.clone()
    best_loss = loss_fct(y_model, y)  
    
    factor = 1.0
    for n in range(k):
        # forward pass
        x_old = x_next.clone()
        xn_var = Variable(x_next, requires_grad=True)
        y_model = model(xn_var)
        loss = loss_fct(y_model, y)

        # compute gradient
        grads = torch.autograd.grad(loss.mean(), xn_var)[0].data
        
        # topk
        signs = torch.gt(grads,0).float()
        grads = (signs - x_next) * grads
        grads -= x * grads
        
        rand_vars = torch.rand(len(grads),len(grads[0]))
        kvals, kidx = grads.topk(k=min(10000, max(1, int(factor))), dim=1)
        x_next.scatter_(dim=1, index=kidx, src=signs.gather(dim=1,index=kidx))

        # projection
        x_next = clip_tensor(x_next)
        x_next = or_float_tensors(x_next, x)
        
        # compute adversarial loss
        loss_adv = loss_fct(model(Variable(x_next)), y)

        found = 0
        for r in range(len(x)):
            if loss_adv.data[r] > best_loss.data[r]:
                x_best[r] = x_next[r].clone()
                best_loss.data[r] = loss_adv.data[r]
                found += 1
        
        #x_next = x_best.clone()
        if found < len(x) * 0.5 and loss.data.mean() > loss_adv.data.mean():
        #if loss.data.mean() > loss_adv.data.mean():
            factor = max(factor * 0.5, 0.25)
            x_next = x_old
            #if found is False:
            if factor < 0.5:
              break
        else:
            factor = factor * 2.0
            #x_next = x_best.clone()

    x_next = x_best
    # compute adversarial loss
    y_model = model(Variable(x_next))
    loss_adv = loss_fct(y_model, y).data
    
    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
            (loss_natural.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))

    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next

def topkrestart(x,
            y,
            model,
            loss_fct,
            k=25,
            epsilon=0.02,
            alpha=0.5,
            is_report_loss_diff=False,
            use_sample=False):
   
    #x_next = topk(x,y,model,loss_fct,k,epsilon,alpha,is_report_loss_diff=False,use_sample=False)

    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)

    # compute natural loss
    y_model = model(Variable(x))
    loss_natural = loss_fct(y_model, y).data

    x_next = x.clone()

    x_best = topk(x,y.data,model,loss_fct,k,epsilon,alpha,False,use_sample)
    y_model = model(Variable(x_best.cuda()))
    best_loss = loss_fct(y_model, y)

    for i in range(10):
        x_rand = torch.rand(len(x),len(x[0]))
        x_rand = torch.lt(x_rand, thresholds).float()
        x_rand = or_float_tensors(x_rand.cpu(), x.cpu()).cuda()
        x_next = topk(x,y.data,model,loss_fct,k,epsilon,alpha,False,use_sample,x_rand)
        y_model = model(Variable(x_next.cuda()))
        loss_adv = loss_fct(y_model, y)

        for r in range(len(x)):
            if loss_adv.data[r] > best_loss.data[r]:
                x_best[r] = x_next[r].clone()
                best_loss.data[r] = loss_adv.data[r]
        #print(best_loss.data.mean())

    x_next = x_best.clone()
    x_next = clip_tensor(x_next)
    x_next = or_float_tensors(x_next.cpu(), x.cpu()).cuda()

    # compute adversarial loss
    y_model = model(Variable(x_next))
    loss_adv = loss_fct(y_model, y).data

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
            (loss_natural.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))

    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next


def topkextra(x,
            y,
            model,
            loss_fct,
            k=25,
            epsilon=0.02,
            alpha=0.5,
            is_report_loss_diff=False,
            use_sample=False):
    
    x_next = topk(x,y,model,loss_fct,k,epsilon,alpha,False,use_sample)
    
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
        x_next = x_next.cuda()
    y = Variable(y)

    # compute natural loss
    y_model = model(Variable(x))
    loss_natural = loss_fct(y_model, y).data

    for i in range(1):
        x_best = x_next.clone()
        y_model = model(Variable(x_next))
        best_loss = loss_fct(y_model, y)
        
        factor = 2*len(x)
        no_improve = 0
        for n in range(k):
            # forward pass
            # x_old = x_next.clone()
            xn_var = Variable(x_next, requires_grad=True)
            y_model = model(xn_var)
            loss = loss_fct(y_model, y)

            # compute gradient
            grads = torch.autograd.grad(loss.mean(), xn_var)[0].data
        
            # topk
            signs = torch.gt(grads,0).float()
            grads = (signs - x_next) * grads
            grads -= x * grads
        
            rand_vars = torch.rand(len(grads),len(grads[0]))
            if next(model.parameters()).is_cuda:
                rand_vars = rand_vars.cuda()
            grads = rand_vars * grads

            kvals, kidx = grads.topk(k=min(10000, max(1, int(factor))), dim=1)
            x_next.scatter_(dim=1, index=kidx, src=signs.gather(dim=1,index=kidx))

            # projection
            x_next = clip_tensor(x_next)
            x_next = or_float_tensors(x_next, x)
        
            # compute adversarial loss
            loss_adv = loss_fct(model(Variable(x_next)), y)
        
            factor = random.random() * 2*len(x) + 1.0
            #if loss.data.mean() > loss_adv.data.mean():
            #    x_next = x_old
            #    factor = max(1,factor * 0.5)
            #else:
            #    factor = factor * 2.0

            #print(loss_adv.data.mean())
    
            found = False
            for r in range(len(x)):
                if loss_adv.data[r] > best_loss.data[r]:
                    x_best[r] = x_next[r].clone()
                    best_loss.data[r] = loss_adv.data[r]
                    found = True
            if found is True:
            # if loss_adv.data.mean() > best_loss.data.mean():
                #x_best = x_next.clone()
                #best_loss = loss_adv
                #x_next = x_best.clone()
                no_improve = 0
            else:
                no_improve += 1
    
            if no_improve > len(x):
                break

        x_next = topk(x,y.data,model,loss_fct,k,epsilon,alpha,False,use_sample,x_best).cuda()
    
    
    # projection
    #x_next = clip_tensor(x_next)
    #x_next = or_float_tensors(x_next, x)
     #x_next = x_best
    
    # compute adversarial loss
    y_model = model(Variable(x_next))
    loss_adv = loss_fct(y_model, y).data
    
    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
            (loss_natural.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))

    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next


def dfgsm_k(x,
            y,
            model,
            loss_fct,
            k=25,
            epsilon=0.02,
            alpha=0.5,
            is_report_loss_diff=False,
            use_sample=False):
    """
    FGSM^k with deterministic rounding
    :param y:
    :param x: (tensor) feature vector
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :param epsilon: update value in each direction
    :param alpha:
    :param is_report_loss_diff:
    :param use_sample:
    :return: the adversarial version of x according to dfgsm_k (tensor)
    """
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # initialize starting point
    x_next = get_x0(x, use_sample)

    # multi-step
    for t in range(k):
        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # compute gradient
        grad_vars = torch.autograd.grad(loss.mean(), x_var)

        # find the next sample
        x_next = x_next + epsilon * torch.sign(grad_vars[0].data)

        # projection
        x_next = clip_tensor(x_next)

    # rounding step
    x_next = round_x(x_next, alpha=alpha)

    # feasible projection
    x_next = or_float_tensors(x_next, x)

    # compute adversarial loss
    loss_adv = loss_fct(model(Variable(x_next)), y).data

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
            (loss_natural.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))

    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next

def flex_rfgsm_k(x, y, model, loss_fct, k=25, epsilon=0.02, is_report_loss_diff=False, use_sample=False):
    """
    FGSM^k with randomized rounding
    :param x: (tensor) feature vector
    :param y:
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :param epsilon: update value in each direction
    :param is_report_loss_diff:
    :param use_sample:
    :return: the adversarial version of x according to rfgsm_k (tensor)
    """
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # initialize starting point
    x_next = get_x0(x, use_sample)

    factor = 1000.0
    for n in range(k):
        x_old = x_next.clone()
        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # compute gradient
        grad_vars = torch.autograd.grad(loss.mean(), x_var)[0].data
        grad_vars -= grad_vars * x

        # find the next sample
        #x_next = x_next + factor * torch.sign(grad_vars[0].data)
        x_next = x_next + factor * grad_vars

        # projection
        x_next = clip_tensor(x_next)

        loss_adv = loss_fct(model(Variable(x_next)), y)
        
        #if is_report_loss_diff:
        #print("Natural loss (%.4f) vs Clean adversarial loss (%.4f) vs. Adversarial loss (%.4f), Difference: (%.4f)" %
        #     (loss_natural.mean(), loss_clean.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))
        print("Natural loss (%.4f) vs. Adversarial loss (%.4f), Difference: (%.4f)" %
             (loss_natural.mean(), loss_adv.data.mean(), loss_adv.mean() - loss_natural.mean()))
        
        if loss.data.mean() > loss_adv.data.mean():
            factor = factor * 0.5
            x_next = x_old
        else:
            factor = factor * 2.0
      
    x_best = x_next.clone()
    best_val = -1.0
    x_old = x_next.clone()
    # rounding step
    for n in range(1000):
        alpha = torch.rand(x_next.size())
        if x_next.is_cuda:
            alpha = alpha.cuda()
        x_next = round_x(x_next, alpha=alpha)
        x_next = or_float_tensors(x_next, x)
        loss_adv = loss_fct(model(Variable(x_next)), y).data
        print(loss_adv.mean())
        if loss_adv.mean() > best_val or best_val == -1.0:
            best_val = loss_adv.mean()
            x_bext = x_next.clone()
        x_next = x_old

    x_next = x_best

    # feasible projection
    x_next = or_float_tensors(x_next, x)

    # compute adversarial loss
    loss_adv = loss_fct(model(Variable(x_next)), y).data

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
          (loss_natural.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))

    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next

def rfgsm_k(x, y, model, loss_fct, k=25, epsilon=0.02, is_report_loss_diff=False, use_sample=False):
    """
    FGSM^k with randomized rounding
    :param x: (tensor) feature vector
    :param y:
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :param epsilon: update value in each direction
    :param is_report_loss_diff:
    :param use_sample:
    :return: the adversarial version of x according to rfgsm_k (tensor)
    """
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # initialize starting point
    x_next = get_x0(x, use_sample)

    # multi-step with gradients
    for t in range(k):
        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # compute gradient
        grad_vars = torch.autograd.grad(loss.mean(), x_var)

        # find the next sample
        x_next = x_next + epsilon * torch.sign(grad_vars[0].data)

        # projection
        x_next = clip_tensor(x_next)

    # rounding step
    alpha = torch.rand(x_next.size())
    if x_next.is_cuda:
        alpha = alpha.cuda()
    x_next = round_x(x_next, alpha=alpha)

    # feasible projection
    x_next = or_float_tensors(x_next, x)

    # compute adversarial loss
    loss_adv = loss_fct(model(Variable(x_next)), y).data

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
          (loss_natural.mean(), loss_adv.mean(), loss_adv.mean() - loss_natural.mean()))

    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next


def bga_k(x, y, model, loss_fct, k=25, is_report_loss_diff=False, use_sample=False):
    """
    Multi-step bit gradient ascent
    :param x: (tensor) feature vector
    :param y:
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :param is_report_loss_diff:
    :param use_sample:
    :return: the adversarial version of x according to bga_k (tensor)
    """
    # some book-keeping
    sqrt_m = torch.from_numpy(np.sqrt([x.size()[1]])).float()

    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
        sqrt_m = sqrt_m.cuda()

    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # keeping worst loss
    loss_worst = loss_natural.clone()
    x_worst = x.clone()

    # multi-step with gradients
    loss = None
    x_var = None
    x_next = None
    for t in range(k):
        if t == 0:
            # initialize starting point
            x_next = get_x0(x, use_sample)
        else:
            # compute gradient
            grad_vars = torch.autograd.grad(loss.mean(), x_var)
            grad_data = grad_vars[0].data

            # compute the updates
            x_update = (sqrt_m * (1. - 2. * x_next) * grad_data >= torch.norm(
                grad_data, 2, 1).unsqueeze(1).expand_as(x_next)).float()

            # find the next sample with projection to the feasible set
            x_next = xor_float_tensors(x_update, x_next)
            x_next = or_float_tensors(x_next, x)

        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # update worst loss and adversarial samples
        replace_flag = (loss.data > loss_worst)
        loss_worst[replace_flag] = loss.data[replace_flag]
        x_worst[replace_flag.unsqueeze(1).expand_as(x_worst)] = x_next[replace_flag.unsqueeze(1)
                                                                       .expand_as(x_worst)]

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
          (loss_natural.mean(), loss_worst.mean(), loss_worst.mean() - loss_natural.mean()))

    if x_worst.is_cuda:
        x_worst = x_worst.cpu()

    return x_worst


def bca_k(x, y, model, loss_fct, k=25, is_report_loss_diff=False, use_sample=False):
    """
    Multi-step bit coordinate ascent
    :param use_sample:
    :param is_report_loss_diff:
    :param y:
    :param x: (tensor) feature vector
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :return: the adversarial version of x according to bca_k (tensor)
    """
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()

    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # keeping worst loss
    loss_worst = loss_natural.clone()
    x_worst = x.clone()

    # multi-step with gradients
    loss = None
    x_var = None
    x_next = None
    for t in range(k):
        if t == 0:
            # initialize starting point
            x_next = get_x0(x, use_sample)
        else:
            # compute gradient
            grad_vars = torch.autograd.grad(loss.mean(), x_var)
            grad_data = grad_vars[0].data

            # compute the updates (can be made more efficient than this)
            aug_grad = (1. - 2. * x_next) * grad_data
            val, _ = torch.topk(aug_grad, 1)
            x_update = (aug_grad >= val.expand_as(aug_grad)).float()

            # find the next sample with projection to the feasible set
            x_next = xor_float_tensors(x_update, x_next)
            x_next = or_float_tensors(x_next, x)

        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # update worst loss and adversarial samples
        replace_flag = (loss.data > loss_worst)
        loss_worst[replace_flag] = loss.data[replace_flag]
        x_worst[replace_flag.unsqueeze(1).expand_as(x_worst)] = x_next[replace_flag.unsqueeze(1)
                                                                       .expand_as(x_worst)]

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
          (loss_natural.mean(), loss_worst.mean(), loss_worst.mean() - loss_natural.mean()))

    if x_worst.is_cuda:
        x_worst = x_worst.cpu()

    return x_worst


def grosse_k(x, y, model, loss_fct, k=25, is_report_loss_diff=False, use_sample=False):
    """
    Multi-step bit coordinate ascent using gradient of output, advancing in direction of maximal change
    :param use_sample:
    :param is_report_loss_diff:
    :param loss_fct:
    :param y:
    :param x: (tensor) feature vector
    :param model: nn model
    :param k: num of steps
    :return adversarial version of x (tensor)
    """

    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()

    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # keeping worst loss
    loss_worst = loss_natural.clone()
    x_worst = x.clone()

    output = None
    x_var = None
    x_next = None
    for t in range(k):
        if t == 0:
            # initialize starting point
            x_next = get_x0(x, use_sample)
        else:
            grad_vars = torch.autograd.grad(output[:, 0].mean(), x_var)
            grad_data = grad_vars[0].data

            # Only consider gradients for points of 0 value
            aug_grad = (1. - x_next) * grad_data
            val, _ = torch.topk(aug_grad, 1)
            x_update = (aug_grad >= val.expand_as(aug_grad)).float()

            # find the next sample with projection to the feasible set
            x_next = xor_float_tensors(x_update, x_next)
            x_next = or_float_tensors(x_next, x)

        x_var = Variable(x_next, requires_grad=True)
        output = model(x_var)

        loss = loss_fct(output, y)

        # update worst loss and adversarial samples
        replace_flag = (loss.data > loss_worst)
        loss_worst[replace_flag] = loss.data[replace_flag]
        x_worst[replace_flag.unsqueeze(1).expand_as(x_worst)] = x_next[replace_flag.unsqueeze(1)
                                                                       .expand_as(x_worst)]

    if is_report_loss_diff:
        print("Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)" %
          (loss_natural.mean(), loss_worst.mean(), loss_worst.mean() - loss_natural.mean()))

    if x_worst.is_cuda:
        x_worst = x_worst.cpu()

    return x_worst

'''
def inner_maximizer(x, y, model, loss_fct, iterations=100, method='natural', mal_index=0, dataset={}):
    """
    A wrapper function for the above algorithim
    :param iterations:
    :param x:
    :param y:
    :param model:
    :param loss_fct:
    :param method: one of 'dfgsm_k', 'rfgsm_k', 'bga_k', 'bca_k', 'natural
    :return: adversarial examples
    """
#print(x)
#for v in range(len(x)):
#    x_next = topk(x[v:v+1,:], y, model, loss_fct, k=iterations)
#    print(torch.sum(x_next,dim=1))
#x_next = dfgsm_k(x, y, model, loss_fct, k=iterations)
#print(torch.sum(x_next,dim=1))
#return x_next
    x_next = x.clone()

    for i in range(len(x)):
        x1 = x[i:i+1,:]
        y1 = y[i:i+1]

        print("\n")
        #dfgsm_k(x, y, model, loss_fct, k=iterations)
        #rfgsm_k(x, y, model, loss_fct, k=iterations)
        #bga_k(x, y, model, loss_fct, k=iterations)
        #bca_k(x, y, model, loss_fct, k=iterations)
        #grosse_k(x, y, model, loss_fct, k=iterations)
        #newloss(x, y, model, loss_fct, k=iterations)
        topk(x1, y1, model, loss_fct, k=iterations, is_report_loss_diff=True)
        topkextra(x1, y1, model, loss_fct, k=iterations, is_report_loss_diff=True)
        print("\n")
    
        if method == 'dfgsm_k':
            x_next[i,:] = dfgsm_k(x1, y1, model, loss_fct, k=iterations)
        elif method == 'rfgsm_k':
            x_next[i,:] = rfgsm_k(x1, y1, model, loss_fct, k=iterations)
        elif method == 'bga_k':
            x_next[i,:] = bga_k(x1, y1, model, loss_fct, k=iterations)
        elif method == 'bca_k':
            x_next[i,:] = bca_k(x1, y1, model, loss_fct, k=iterations)
        elif method == 'grosse':
            x_next[i,:] = grosse_k(x1, y1, model, loss_fct, k=iterations)
        elif method == 'matching':
            x_next[i,:] = matching(x1, y1, model, loss_fct, k=iterations, mal_index = mal_index, dataset = dataset)
        elif method == 'topk':
            x_next[i,:] = topk(x1, y1, model, loss_fct, is_report_loss_diff=True, k=iterations)
        elif method == 'topk+':
            x_next[i,:] = topkextra(x1, y1, model, loss_fct, is_report_loss_diff=True, k=iterations)
        elif method == 'topkr':
            x_next[i,:] = topkrestart(x1, y1, model, loss_fct, is_report_loss_diff=True, k=iterations)
        elif method == 'myloss':
            x_next[i,:] = newloss(x1, y1, model, loss_fct, k=iterations)
        elif method == 'natural':
            return x
        else:
            raise Exception('No such inner maximizer algorithm')
    return x_next

'''
def inner_maximizer(x, y, model, loss_fct, iterations=100, method='natural', mal_index=0, dataset={}):
    """
    A wrapper function for the above algorithim
    :param iterations:
    :param x:
    :param y:
    :param model:
    :param loss_fct:
    :param method: one of 'dfgsm_k', 'rfgsm_k', 'bga_k', 'bca_k', 'natural
    :return: adversarial examples
    """

    #print("\n")
    #dfgsm_k(x, y, model, loss_fct, k=iterations)
    #rfgsm_k(x, y, model, loss_fct, k=iterations)
    #bga_k(x, y, model, loss_fct, k=iterations)
    #bca_k(x, y, model, loss_fct, k=iterations)
    #grosse_k(x, y, model, loss_fct, k=iterations)
    #newloss(x, y, model, loss_fct, k=iterations)
    #topk(x, y, model, loss_fct, k=iterations, is_report_loss_diff=True)
    #topkextra(x, y, model, loss_fct, k=iterations, is_report_loss_diff=True)
    #topkrestart(x, y, model, loss_fct, k=iterations, is_report_loss_diff=True)
    #print("\n")

    if method == 'rand':
        method = random.choice(["rfgsm_k", "bca_k", "topk", "topk+", "topkr", "topkm", "topk+", "dfgsm_k" ])
    
    if method == 'dfgsm_k':
        return dfgsm_k(x, y, model, loss_fct, is_report_loss_diff=True, k=iterations)
    elif method == 'rfgsm_k':
        return rfgsm_k(x, y, model, loss_fct, is_report_loss_diff=True, k=iterations)
    elif method == 'bga_k':
        return bga_k(x, y, model, loss_fct, is_report_loss_diff=True, k=iterations)
    elif method == 'bca_k':
        return bca_k(x, y, model, loss_fct, is_report_loss_diff=True, k=iterations)
    elif method == 'grosse':
        return grosse_k(x, y, model, loss_fct, is_report_loss_diff=True,  k=iterations)
    elif method == 'topk':
        return topk(x, y, model, loss_fct, is_report_loss_diff=True, k=iterations)
    elif method == 'topk+':
        return topkextra(x, y, model, loss_fct, is_report_loss_diff=True, k=iterations)
    elif method == 'topkr':
        return topkrestart(x, y, model, loss_fct, is_report_loss_diff=True, k=iterations)
    elif method == 'natural':
        return x
    else:
        raise Exception('No such inner maximizer algorithm')
