from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import Voronoi
from skimage.draw import polygon
import random

def voronoi(points, size):
    # Add points at edges to eliminate infinite ridges
    edge_points = size*np.array([[-1, -1], [-1, 2], [2, -1], [2, 2]])
    new_points = np.vstack([points, edge_points])
    
    # Calculate Voronoi tessellation
    vor = Voronoi(new_points)    
    return vor

width = height = size = 3600
n = 384

x = np.random.randint(0, width, (n,)) #随机采样n个点
y = np.random.randint(0, height, (n,))

points = np.vstack([x, y]).T #x和y依次组成n个点[[x, y],...]

cyan = (12, 236, 221)
yellow = (255, 243, 56)
pink = (196, 0, 255)
magenta = (255, 103, 231)

new_points = points.copy()

for k in range(100):
    vor = voronoi(new_points, size)

    im = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    for i, point in enumerate(new_points):
        circle_size = np.array([20, 20])
        p1 = point - circle_size/2
        p2 = point + circle_size/2
        draw.ellipse((*p1, *p2), fill=(0, 0, 255))

    new_points = []

    for i, region in enumerate(vor.regions):
        if len(region) == 0 or -1 in region: continue
        poly = np.array([vor.vertices[i] for i in region])
        
        for i in range(len(poly)):
            draw.line((*poly[-i], *poly[-i-1]), fill=(50, 50 ,50), width=5)

        center = poly.mean(axis=0)
        new_points.append(center)

        circle_size = np.array([20, 20])
        p1 = center - circle_size/2
        p2 = center + circle_size/2
        draw.ellipse((*p1, *p2), fill=(255, 0, 0))

    new_points = np.array(new_points).clip(0, size)

    im = im.crop((200, 1000, 3400, 2600))
    im = im.resize((800, 400), resample=Image.ANTIALIAS)
    im.save(f"figure_4/out_{k}.png")
