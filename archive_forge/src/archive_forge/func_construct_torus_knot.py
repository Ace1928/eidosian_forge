import numpy as np
from panda3d.core import (
from panda3d.core import RenderModeAttrib
def construct_torus_knot(p=2, q=3, num_vertices=100, radius=1.0):
    """
    Construct a torus knot geometry using a structured array approach for vertex and color data management.
    A torus knot is a type of knot that lies on the surface of a torus in three-dimensional space.
    The parameters 'p' and 'q' are integers that determine the type of torus knot.
    The 'num_vertices' parameter specifies the number of vertices to be used in constructing the knot.
    The 'radius' parameter specifies the scale of the torus knot.

    Parameters:
        p (int): Number of times the knot wraps around the torus tube.
        q (int): Number of times the knot goes around the torus center.
        num_vertices (int): Number of vertices to generate for the knot.
        radius (float): Scale factor for the size of the torus knot.

    Returns:
        GeomNode: A geometry node containing the constructed torus knot geometry.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData('torus_knot_vertices_and_colors', vertex_format, Geom.UHStatic)
    vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
    color_writer = GeomVertexWriter(vertex_data, 'color')
    vertices = np.zeros((num_vertices, 3), dtype=np.float32)
    colors = np.zeros((num_vertices, 4), dtype=np.float32)
    angle_increment = 2 * np.pi / num_vertices
    tube_radius = 0.1 * radius
    for i in range(num_vertices):
        phi = i * angle_increment * p
        theta = i * angle_increment * q
        x = (radius + tube_radius * np.cos(q * phi)) * np.cos(p * phi)
        y = (radius + tube_radius * np.cos(q * phi)) * np.sin(p * phi)
        z = tube_radius * np.sin(q * phi)
        vertices[i] = [x, y, z]
        colors[i] = [np.sin(phi), np.cos(phi), np.abs(np.sin(theta)), 1.0]
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)
    triangle_indices = np.array([(i, (i + 1) % num_vertices, (i + 2) % num_vertices) for i in range(num_vertices)], dtype=np.int32)
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode('torus_knot_geom_node')
    geometry_node.addGeom(geometry)
    return geometry_node