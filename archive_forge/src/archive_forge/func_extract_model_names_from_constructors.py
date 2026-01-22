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
def extract_model_names_from_constructors(self):
    """
        Extract and return a list of model names by inspecting the available constructor methods within the class.
        Utilizes numpy array for storage and manipulation of the model names.
        """
    constructor_methods = dir(self)
    model_names = np.array([method_name[10:] for method_name in constructor_methods if method_name.startswith('construct_')])
    return model_names