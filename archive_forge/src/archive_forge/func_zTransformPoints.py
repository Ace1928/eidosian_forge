from math import cos, sin, tan, radians
def zTransformPoints(matrix, V):
    return list(map(lambda x, matrix=matrix: zTransformPoint(matrix, x), V))