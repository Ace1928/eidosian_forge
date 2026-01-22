import numpy as np
from panda3d.core import (
from panda3d.core import RenderModeAttrib
def construct_pyramid():
    """
    Construct a pyramid using vertex data and geom nodes, meticulously defining each vertex and its corresponding color.
    This function utilizes a structured array approach for optimal data management and efficiency.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData('pyramid_vertices_and_colors', vertex_format, Geom.UHStatic)
    vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
    color_writer = GeomVertexWriter(vertex_data, 'color')
    vertices = np.array([[0, 0, 1], [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
    colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]], dtype=np.float32)
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)
    triangle_indices = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [1, 3, 4]], dtype=np.int32)
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode('pyramid_geom_node')
    geometry_node.addGeom(geometry)
    return geometry_node