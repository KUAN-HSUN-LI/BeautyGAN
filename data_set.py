from param import *
from PIL import Image
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt


def get_fileNames(path):
    filenames = []
    for parent, dirnames, files in os.walk(path):
        for name in files:
            filenames.append(os.path.join(parent, name))
    return filenames


def cut_dataName(kind, mode):
    seed = get_random_seed()
    assert isinstance(seed, int)
    np.random.seed(seed)
    if mode == 'makeup':
        fileNames = get_fileNames('./all/'+kind+'/makeup/')
        np.random.shuffle(fileNames)
        test = fileNames[:250]
        fileNames = fileNames[250::]
    elif mode == 'Nmakeup':
        fileNames = get_fileNames('./all/'+kind+'/non-makeup/')
        np.random.shuffle(fileNames)
        test = fileNames[:100]
        fileNames = fileNames[100::]
    cut = int(len(fileNames) * 0.1)
    valid = fileNames[:cut]
    train = fileNames[cut::]

    return test, valid, train


def get_images(mode, data):
    assert data in ['test', 'valid', 'train']
    test, valid, train = cut_dataName('images', mode)
    imgs = []
    if data == 'test':
        for filename in test:
            img = np.array(Image.open(filename))
            img = get_transform(img)
            imgs.append(img)
    elif data == 'valid':
        for filename in valid:
            img = np.array(Image.open(filename))
            img = get_transform(img)
            imgs.append(img)
    elif data == 'train':
        for filename in train:
            img = np.array(Image.open(filename))
            img = get_transform(img)
            imgs.append(img)
    return (np.array(imgs, dtype=np.float32) - 127.5) / 127.5


def get_segs(mode, data, feature):
    assert data in ['test', 'valid', 'train']
    test, valid, train = cut_dataName('segs', mode)
    segs = []
    if data == 'test':
        for filename in test:
            img = np.array(Image.open(filename))
            img = img[:, :, np.newaxis]
            img = np.tile(img, 3)
            seg = get_feature(img, feature)
            seg = get_transform(seg)
            seg[seg >= 0.5] = 1
            seg[seg < 0.5] = 0
            segs.append(seg)
    elif data == 'valid':
        for filename in valid:
            img = np.array(Image.open(filename))
            img = img[:, :, np.newaxis]
            img = np.tile(img, 3)
            seg = get_feature(img, feature)
            seg = get_transform(seg)
            seg[seg >= 0.5] = 1
            seg[seg < 0.5] = 0
            segs.append(seg)
    elif data == 'train':
        for filename in train:
            img = np.array(Image.open(filename))
            img = img[:, :, np.newaxis]
            img = np.tile(img, 3)
            seg = get_feature(img, feature)
            seg = get_transform(seg)
            seg[seg >= 0.5] = 1
            seg[seg < 0.5] = 0
            segs.append(seg)
    return np.array(segs)


def get_feature(img, feature):
    seg = np.zeros([321, 321, 3])
    for x in feature:
        seg += (img == x)
    return seg


def get_transform(img):
    img = cv2.resize(img, (256, 256))
    return img


class img_gen():
    def __init__(self, mode, data, batch_size):
        self.images = get_images(mode, data)
        self.batch_size = batch_size

    def data_gen(self):
        while True:
            for i in range(0, len(self.images), self.batch_size):
                yield self.images[i:i+1]


class seg_gen():
    def __init__(self, mode, data, feature, batch_size):
        self.segs = get_segs(mode, data, feature)
        self.batch_size = batch_size

    def data_gen(self):
        while True:
            for i in range(0, len(self.segs), self.batch_size):
                yield self.segs[i:i+1]
