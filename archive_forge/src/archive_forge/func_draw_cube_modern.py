import math
import ctypes
import pygame as pg
def draw_cube_modern(shader_data, filled_cube_indices, outline_cube_indices, rotation):
    """
    Draw a cube in the 'modern' Open GL style, for post 3.1 versions of
    open GL.

    :param shader_data: compile vertex & pixel shader data for drawing a cube.
    :param filled_cube_indices: the indices to draw the 'filled' cube.
    :param outline_cube_indices: the indices to draw the 'outline' cube.
    :param rotation: the current rotations to apply.
    """
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glDisable(GL.GL_BLEND)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
    GL.glUniform4f(shader_data['constants']['colour_mul'], 1, 1, 1, 1)
    GL.glUniform4f(shader_data['constants']['colour_add'], 0, 0, 0, 0.0)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, shader_data['buffer']['filled'])
    GL.glDrawElements(GL.GL_TRIANGLES, len(filled_cube_indices), GL.GL_UNSIGNED_INT, None)
    GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
    GL.glEnable(GL.GL_BLEND)
    GL.glUniform4f(shader_data['constants']['colour_mul'], 0, 0, 0, 0.0)
    GL.glUniform4f(shader_data['constants']['colour_add'], 1, 1, 1, 1.0)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, shader_data['buffer']['outline'])
    GL.glDrawElements(GL.GL_LINES, len(outline_cube_indices), GL.GL_UNSIGNED_INT, None)
    rotation.phi += 1.0
    model = eye(4, dtype=float32)
    rotate(model, rotation.phi, 0, 1, 0)
    rotate(model, rotation.psi, 1, 0, 0)
    GL.glUniformMatrix4fv(shader_data['constants']['model'], 1, False, model)