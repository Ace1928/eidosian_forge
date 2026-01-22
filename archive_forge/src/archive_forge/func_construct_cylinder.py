import numpy as np
from panda3d.core import (
from panda3d.core import RenderModeAttrib
def construct_cylinder(num_segments=32, height=1.0, radius=1.0):
    """
    Construct a cylinder using vertex data and geom nodes, meticulously defining each vertex and its corresponding color.
    This function utilizes a structured array approach for optimal data management and efficiency.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData('cylinder_vertices_and_colors', vertex_format, Geom.UHStatic)
    vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
    color_writer = GeomVertexWriter(vertex_data, 'color')
    vertices = np.zeros((num_segments * 2, 3), dtype=np.float32)
    colors = np.zeros((num_segments * 2, 4), dtype=np.float32)
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices[i] = [x, y, 0]
        vertices[i + num_segments] = [x, y, height]
        colors[i] = [1, 0, 0, 1]
        colors[i + num_segments] = [0, 0, 1, 1]
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)
    triangle_indices = np.zeros((num_segments * 2, 3), dtype=np.int32)
    for i in range(num_segments):
        triangle_indices[i] = [i, (i + 1) % num_segments, i + num_segments]
        triangle_indices[i + num_segments] = [(i + 1) % num_segments, i + num_segments, (i + 1) % num_segments + num_segments]
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode('cylinder_geom_node')
    geometry_node.addGeom(geometry)
    return geometry_node