import math
import ctypes
import pygame as pg
def frustum(left, right, bottom, top, znear, zfar):
    """
    Build a perspective matrix from the clipping planes, or camera 'frustrum'
    volume.

    :param left: left position of the near clipping plane.
    :param right: right position of the near clipping plane.
    :param bottom: bottom position of the near clipping plane.
    :param top: top position of the near clipping plane.
    :param znear: z depth of the near clipping plane.
    :param zfar: z depth of the far clipping plane.

    :return: A perspective matrix.
    """
    perspective_matrix = zeros((4, 4), dtype=float32)
    perspective_matrix[0, 0] = +2.0 * znear / (right - left)
    perspective_matrix[2, 0] = (right + left) / (right - left)
    perspective_matrix[1, 1] = +2.0 * znear / (top - bottom)
    perspective_matrix[3, 1] = (top + bottom) / (top - bottom)
    perspective_matrix[2, 2] = -(zfar + znear) / (zfar - znear)
    perspective_matrix[3, 2] = -2.0 * znear * zfar / (zfar - znear)
    perspective_matrix[2, 3] = -1.0
    return perspective_matrix