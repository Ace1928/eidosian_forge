import math
def facet_stl(triangle):
    vertex1, vertex2, vertex3 = triangle
    a = (vertex3[0] - vertex1[0], vertex3[1] - vertex1[1], vertex3[2] - vertex1[2])
    b = (vertex2[0] - vertex1[0], vertex2[1] - vertex1[1], vertex2[2] - vertex1[2])
    normal = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    return ''.join(['  facet normal %f %f %f\n' % tuple(normal), '    outer loop\n', '      vertex %f %f %f\n' % tuple(vertex1), '      vertex %f %f %f\n' % tuple(vertex2), '      vertex %f %f %f\n' % tuple(vertex3), '    endloop\n', '  endfacet\n'])