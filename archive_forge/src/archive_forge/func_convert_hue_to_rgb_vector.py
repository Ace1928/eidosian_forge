import sys
import logging
import math
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, VBase4, AmbientLight, DirectionalLight, ColorAttrib
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter
from panda3d.core import GeomTriangles, GeomNode
from direct.task import Task
from models import (
def convert_hue_to_rgb_vector(hue_angle_degrees):
    """
    Convert a hue value (in degrees) to an RGB color vector with full saturation and brightness.
    This function meticulously calculates the RGB values based on the hue angle provided, ensuring
    the output is a tuple of RGB values, each component ranging from 0 to 1.

    Parameters:
    - hue_angle_degrees (float): The hue angle in degrees, which will be normalized to a range of 0-360.

    Returns:
    - tuple: A tuple representing the RGB color (r, g, b), each component as a float from 0 to 1.
    """
    normalized_hue_angle = hue_angle_degrees % 360
    maximum_saturation = 1.0
    maximum_brightness = 1.0
    intermediate_x = 1 - abs(normalized_hue_angle / 60.0 % 2 - 1)
    base_rgb_adjustment = 0.0
    rgb_sector_matrix = np.array([(maximum_saturation, intermediate_x, 0), (intermediate_x, maximum_saturation, 0), (0, maximum_saturation, intermediate_x), (0, intermediate_x, maximum_saturation), (intermediate_x, 0, maximum_saturation), (maximum_saturation, 0, intermediate_x)], dtype=np.float32)
    sector_index = int(normalized_hue_angle // 60)
    sector_index_int = int(sector_index)
    rgb_values = np.take(rgb_sector_matrix, sector_index_int, axis=0)
    adjusted_rgb_values = np.add(rgb_values, base_rgb_adjustment)
    return tuple(adjusted_rgb_values)