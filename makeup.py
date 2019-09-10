import numpy as np

def mask(img, seg, feature):
    seg = seg == feature
    img = img * seg
    img[img == 0] = np.inf

