from __future__ import absolute_import
from __future__ import division
import pythreejs
import os
import time
import warnings
import tempfile
import uuid
import base64
from io import BytesIO as StringIO
import six
import numpy as np
import PIL.Image
import matplotlib.style
import ipywidgets
import IPython
from IPython.display import display
import ipyvolume as ipv
import ipyvolume.embed
from ipyvolume import utils
from . import ui
def material_phong(emissive='#000000', specular='#111111', shininess=30, flat_shading=False, opacity=1, transparent=False, **kwargs):
    """Sets the current material to a :any:`pythreejs.MeshPhongMaterial`.

    :param color emissive: Emissive (light) color of the material, essentially a solid color unaffected by other lighting. Default is black.
    :param color specular: Specular color of the material. Default is a Color set to 0x111111 (very dark grey). This defines how shiny the material is and the color of its shine.
    :param snininess: How shiny the specular highlight is; a higher value gives a sharper highlight. Default is 30.
    :param flat_shading: A technique for color computing where all polygons reflect as a flat surface. Default False
    :param float opacity: {opacity}
    :param bool transparent: {transparent}
    :param kwargs: Arguments passed on the constructor of :any:`pythreejs.MeshPhongMaterial`
    :return: :any:`pythreejs.MeshPhongMaterial`
    """
    material = pythreejs.MeshPhongMaterial(emissive=emissive, specular=specular, shininess=shininess, flat_shading=flat_shading, opacity=opacity, transparent=transparent, side=pythreejs.enums.Side.DoubleSide, **kwargs)
    current.material = material
    return current.material