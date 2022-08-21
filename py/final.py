import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
from skimage.draw import polygon
from PIL import Image
from noise import snoise3

size = 1024
n = 256
map_seed = 762345

np.random.seed(map_seed)

def voronoi(points, size):
    # Add points at edges to eliminate infinite ridges
    edge_points = size*np.array([[-1, -1], [-1, 2], [2, -1], [2, 2]])
    new_points = np.vstack([points, edge_points])
    
    # Calculate Voronoi tessellation
    vor = Voronoi(new_points)
    
    return vor

def voronoi_map(vor, size):
    # Calculate Voronoi map
    vor_map = np.zeros((size, size), dtype=np.uint32)

    for i, region in enumerate(vor.regions):
        # Skip empty regions and infinte ridge regions
        if len(region) == 0 or -1 in region: continue
        # Get polygon vertices
        # 看文档，一个region用一个索引数组表示，如
        # 而这些索引是用来所以vor.vertices，
        # vor.vertices是新生成的点，不是输入点。
        # np.array([vor.vertices[i][::-1] for i in region]得到是顶点坐标数组，这些顶点围成region。
        # T一下，x，y分别存放这些顶点的x坐标（数组）和y坐标（数组）
        x, y = np.array([vor.vertices[i][::-1] for i in region]).T
        # Get pixels inside polygon
        # 获取位于指定多边形内的所有像素点，返回row坐标数组和column坐标数组
        rr, cc = polygon(x, y)
        # Remove pixels out of image bounds
        in_box = np.where((0 <= rr) & (rr < size) & (0 <= cc) & (cc < size))
        rr, cc = rr[in_box], cc[in_box]
        # Paint image
        # 映射：每个像素属于哪个region
        vor_map[rr, cc] = i

    return vor_map

points = np.random.randint(0, size, (514, 2))
vor = voronoi(points, size)
vor_map = voronoi_map(vor, size)

fig = plt.figure(dpi=150, figsize=(4, 4))
plt.scatter(*points.T, s=1)

def relax(points, size, k=10):  
    new_points = points.copy()
    for _ in range(k):
        vor = voronoi(new_points, size)
        new_points = []
        for i, region in enumerate(vor.regions):
            if len(region) == 0 or -1 in region: continue
            poly = np.array([vor.vertices[i] for i in region])
            center = poly.mean(axis=0)
            new_points.append(center)
        new_points = np.array(new_points).clip(0, size)
    return new_points

points = relax(points, size, k=100)
vor = voronoi(points, size)
vor_map = voronoi_map(vor, size)

fig = plt.figure(dpi=150, figsize=(4, 4))
plt.scatter(*points.T, s=1)
# plt.savefig('final/fuck1.png')

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

# 这里我如此理解：
# noise_map返回二维噪声，每个数值在[-1, 1]。np.dstack在两个数组上，一一对应地，进行级联，组成一个新二维数组，每个元素是分别来自两个数组的二元组[x,y]。查看test.py
boundary_displacement = 8
boundary_noise = np.dstack([noise_map(size, 32, 200, octaves=8), noise_map(size, 32, 250, octaves=8)])
# np.indices，看文档或test.py吧，不知道怎么描述。aa=np.indices((size, size)).T刚好就是，aa[x][y] = [x, y]
# 到这里，boundary_noise[x][y] = [noise_x, noise_y]，boundary_displacement*boundary_noise表示该坐标的新坐标的偏移量，偏移范围8。
# 再加上np.indices((size, size)).T，则表示该坐标选择的新坐标点，需要将该新坐标的cell赋值给该坐标。
boundary_noise = np.indices((size, size)).T + boundary_displacement*boundary_noise
# 这里就是防止选择的新坐标超出范围。
boundary_noise = boundary_noise.clip(0, size-1).astype(np.uint32)

blurred_vor_map = np.zeros_like(vor_map)

for x in range(size):
    for y in range(size):
        j, i = boundary_noise[x, y]
        blurred_vor_map[x, y] = vor_map[i, j]

fig, axes = plt.subplots(1, 2)
fig.set_dpi(150)
fig.set_size_inches(8, 4)
axes[0].imshow(vor_map)
axes[1].imshow(blurred_vor_map)

plt.savefig('final/fuck2.png')
