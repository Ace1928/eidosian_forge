import numpy as np
from panda3d.core import (
from panda3d.core import RenderModeAttrib
def construct_sphere(num_segments=32, num_rings=16, radius=1.0):
    """
    Construct a sphere using vertex data and geom nodes, meticulously defining each vertex and its corresponding color.
    This function utilizes a structured array approach for optimal data management and efficiency.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData('sphere_vertices_and_colors', vertex_format, Geom.UHStatic)
    vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
    color_writer = GeomVertexWriter(vertex_data, 'color')
    vertices = np.zeros((num_segments * (num_rings - 1) + 2, 3), dtype=np.float32)
    colors = np.zeros((num_segments * (num_rings - 1) + 2, 4), dtype=np.float32)
    vertices[0] = [0, 0, -radius]
    colors[0] = [1, 0, 0, 1]
    vertices[-1] = [0, 0, radius]
    colors[-1] = [0, 0, 1, 1]
    index = 1
    for j in range(1, num_rings):
        phi = np.pi * j / num_rings
        for i in range(num_segments):
            theta = 2 * np.pi * i / num_segments
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            vertices[index] = [x, y, z]
            colors[index] = [0, 1, 0, 1]
            index += 1
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)
    triangle_indices = []
    for i in range(num_segments):
        triangle_indices.append([0, i + 1, (i + 1) % num_segments + 1])
    for j in range(1, num_rings - 1):
        for i in range(num_segments):
            current = (j - 1) * num_segments + i + 1
            next = current + num_segments
            triangle_indices.append([current, current % num_segments + 1, next])
            triangle_indices.append([next, current % num_segments + 1, next % num_segments + 1])
    top_index = num_segments * (num_rings - 1) + 1
    for i in range(num_segments):
        current = (num_rings - 2) * num_segments + i + 1
        triangle_indices.append([current, current % num_segments + 1, top_index])
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode('sphere_geom_node')
    geometry_node.addGeom(geometry)
    return geometry_node
    color_writer.addData4f(*color)
    triangle_indices = np.array([], dtype=np.int32)
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode('torus_geom_node')
    geometry_node.addGeom(geometry)
    return geometry_node