from random import seed
from tkinter import Y
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
from skimage.draw import polygon
from PIL import Image
from noise import snoise3
from PIL import Image, ImageDraw
import math

size = 1024
size = 500
n = 256
map_seed = 762345

np.random.seed(map_seed)

def noise_map(size, res, seed, octaves=1, persistence=0.5, lacunarity=2.0):
    scale = size/res
    return np.array([[
        snoise3(
            (x+0.1)/scale,
            y/scale,
            seed+map_seed,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity
        )
        for x in range(size)]
        for y in range(size)
    ])


def show_noise(size, res, seed, octaves, persistence, lacunarity):
    out = noise_map(size, res, seed, octaves, persistence, lacunarity)

    im = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    for x in range(size):
        for y in range(size):
            f = int(255 * (out[x][y] + 1) * 0.5)
            if f < 0: f = 0
            if f > 255: f = 255
            draw.point((x, y), fill=(f, f, f))
    
    im.save(f"figure_5/out_res={res}_octaves={octaves}_persistence={persistence}_lacunarity={lacunarity}.png")

#for res in [4, 8, 16, 32, 64]:
for res in [16]:
    #for octaves in range(1, 11):
    for octaves in [1]:
        #for persistence in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]: #不知为啥，没什么变化
        for persistence in [1.0]:
            #for lacunarity in range(1, 11): #不知为啥，没什么变化
            for lacunarity in [5]:
                show_noise(size, res, map_seed, octaves, persistence, lacunarity)
