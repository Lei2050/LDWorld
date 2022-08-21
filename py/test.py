import numpy as np

#a = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

# points = np.random.randint(0, 1024, (514, 2))
# print(points)
# print(len(points))

a = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

b = [[10, 20, 30],
     [40, 50, 60],
     [70, 80, 90]]

dstack = np.dstack([a, b])
print(dstack)
print("=======================")

aaa = np.indices((3, 3))
print(aaa.T)
print("=======================")

boundary_noise = aaa.T + 8 * dstack
print(boundary_noise)
print("=======================")
boundary_noise = boundary_noise.clip(0, 3-1).astype(np.uint32)
print(boundary_noise)
