#taken from https://github.com/SherlockLiao/Deep-Dream

import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
import numpy as np


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def returnImg(a):
    mean = np.array([0.485]).reshape([1, 1, 1])
    std = np.array([0.229]).reshape([1, 1, 1])
    inp = a[0, :, :]
    #inp = inp.transpose(1, 2, 0)
    #inp = std * inp + mean
    inp *= 255
    inp = np.uint8(np.clip(inp, 0, 255))
    #showarray(inp)
    #clear_output(wait=True)
    return inp