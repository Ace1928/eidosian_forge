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
def load_specified_model_by_name(self, model_name):
    """
        Instantiate and display a specific 3D model based on the provided model name.
        Ensures the model is correctly parented to the rendering scene.
        """
    model_constructor_method = getattr(self, f'construct_{model_name}')
    instantiated_model = model_constructor_method()
    instantiated_model.reparentTo(self.render)