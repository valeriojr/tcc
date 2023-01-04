import numpy
import matplotlib.pyplot as pyplot

n = 8
v = n ** 2

vertices = numpy.array([(i, j) for i in range(n) for j in range(n)])
edges = numpy.zeros((v, v))
for i in range(n):
    for j in range(n):
        vi = i * n + j

        if i > 0:
            edges[vi, (i - 1) * n + j] += 1
        if i < n - 1:
            edges[vi, (i + 1) * n + j] += 1
        if j > 0:
            edges[vi, i * n + (j - 1)] += 1
        if j < n - 1:
            edges[vi, i * n + (j + 1)] += 1

edges[0, 1] = 10

points = vertices.reshape((n, n, 2))
pyplot.scatter(points[:, :, 0], points[:, :, 1])
for i in range(v):
    for j in range(v):
        if edges[i, j] > 0:
            pyplot.plot([vertices[i][0], vertices[j][0]], [vertices[i][1], vertices[j][1]], c='k', alpha=10*edges[i, j] / edges.sum())
pyplot.show()
