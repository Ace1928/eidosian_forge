import numpy as np
from panda3d.core import (
from panda3d.core import RenderModeAttrib
def construct_cone(num_segments=32, height=1.0, radius=1.0):
    """
    Construct a cone using vertex data and geom nodes, meticulously defining each vertex and its corresponding color.
    This function utilizes a structured array approach for optimal data management and efficiency.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData('cone_vertices_and_colors', vertex_format, Geom.UHStatic)
    vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
    color_writer = GeomVertexWriter(vertex_data, 'color')
    vertices = np.zeros((num_segments + 1, 3), dtype=np.float32)
    colors = np.zeros((num_segments + 1, 4), dtype=np.float32)
    vertices[0] = [0, 0, height]
    colors[0] = [1, 0, 0, 1]
    for i in range(1, num_segments + 1):
        angle = 2 * np.pi * i / num_segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices[i] = [x, y, 0]
        colors[i] = [0, 1, 0, 1]
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)
    triangle_indices = np.zeros((num_segments, 3), dtype=np.int32)
    for i in range(num_segments):
        triangle_indices[i] = [0, i + 1, (i + 1) % num_segments + 1]
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode('cone_geom_node')
    geometry_node.addGeom(geometry)
    return geometry_node