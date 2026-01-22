import numpy as np
from panda3d.core import (
from panda3d.core import RenderModeAttrib
def construct_square_sheet_with_vertex_data():
    """
    Construct a square sheet using vertex data and geom nodes, with a focus on structured data management.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData('square_sheet_vertices_and_colors', vertex_format, Geom.UHStatic)
    vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
    color_writer = GeomVertexWriter(vertex_data, 'color')
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]], dtype=np.float32)
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)
    triangle_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode('square_sheet_geom_node')
    geometry_node.addGeom(geometry)
    return geometry_node