from math import pi, sqrt
from ipywidgets import register
from .._base.Three import ThreeWidget
from .Object3D_autogen import Object3D as Object3DBase
def lookAt(self, vector):
    self.exec_three_obj_method('lookAt', vector)