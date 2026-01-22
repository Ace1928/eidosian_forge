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
def create_light(self):
    """
        Create the light based on its type and set its color.
        """
    if self.light_type == 'point':
        self.light = PointLight('point_light')
    elif self.light_type == 'ambient':
        self.light = AmbientLight('ambient_light')
    elif self.light_type == 'directional':
        self.light = DirectionalLight('directional_light')
    self.light.setColor(self.color)