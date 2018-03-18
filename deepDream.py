
#from https://github.com/SherlockLiao/Deep-Dream/blob/master/deepdream.py

import numpy as np
import torch
from dreamUtil import returnImg
import scipy.ndimage as nd
from torch.autograd import Variable
import torch.nn as nn


def objective_L2(dst, guide_features):
    return dst.data


def make_step(img, model, control=None, distance=objective_L2):
    #mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    #std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    
    mean = np.array([150]).reshape([1, 1, 1])
    std = np.array([0.229]).reshape([1, 1, 1])

    learning_rate = 0.6
    max_jitter = 32
    num_iterations = 20
    show_every = 10
    end_layer = 3
    guide_features = control

    for i in range(num_iterations):
        shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        img = np.roll(np.roll(img, shift_x, -1), shift_y, -2)
        # apply jitter shift
        model.zero_grad()
        img_tensor = torch.Tensor(img)
        if torch.cuda.is_available():
            img_variable = Variable(img_tensor.cuda(), requires_grad=True)
        else:
            img_variable = Variable(img_tensor, requires_grad=True)

        act_value = model.forward(img_variable)
        diff_out = distance(act_value, guide_features)
        act_value.backward(diff_out)
        ratio = np.abs(img_variable.grad.data.cpu().numpy()).mean()
        learning_rate_use = learning_rate / ratio
        print ("ratio", ratio)
        print ("learning_rate_use", learning_rate_use)
        img_variable.data.add_(img_variable.grad.data * learning_rate_use)
        img = img_variable.data.cpu().numpy()  # b, c, h, w
        img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)
        print ("imgBefore", img)
        #bias = np.mean(img)
        img[0,:,:] = np.clip(img[0,:,:], 0, 255)
        print ("imgAfter", img.shape)
        '''if i == 0 or (i + 1) % show_every == 0:
            showtensor(img)'''
    return img


def dream(model,
          base_img,
          octave_n=6,
          octave_scale=1.4,
          control=None,
          distance=objective_L2,layerNumber = 3):
    
    layerRemoval = 2 + (4 - layerNumber)
    children = list(model.children())[-layerRemoval:-1]
    model = nn.Sequential(*list(model.children())[:-layerRemoval])
    
    print ("model after removing", len(list(model.children())))
    
    octaves = [base_img]
    for i in range(octave_n - 1):
        octaves.append(
            nd.zoom(
                octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale),
                order=1))
        

    #print ("shape of octave", octaves[-1].shape)
    detail = np.zeros_like(octaves[-1])
    #print ("shape of detail", detail.shape)
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            #print ("shape of detail", detail.shape)
            #print ("h1 ", h1, "w1", w1)
            detail = nd.zoom(
                detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        input_oct = octave_base + detail
        print(input_oct.shape)
        out = make_step(input_oct, model, control, distance=distance)
        #print("shape of out", out.shape)
        detail = out - octave_base
    print ("out")
    print (out)
    #showtensor(out)
    #out = returnImg(out)
    model = nn.Sequential(*list(model.children()) + children)
    print ("model after adding", len(list(model.children())))
    return out