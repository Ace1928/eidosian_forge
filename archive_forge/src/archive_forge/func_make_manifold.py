import tkinter
import math
import sys
import time
from tkinter import ttk
from . import gui_utilities
from .gui_utilities import UniformDictController, FpsLabelUpdater
from .view_scale_controller import ViewScaleController
from .raytracing_view import *
from .geodesics_window import GeodesicsWindow
from .hyperboloid_utilities import unit_3_vector_and_distance_to_O13_hyperbolic_translation
from .zoom_slider import Slider, ZoomSlider
def make_manifold(self):
    for f in self.filling_dict['fillings'][1]:
        m, l = f
        m = round(m)
        l = round(l)
        g = abs(_gcd(m, l))
        if g != 0:
            m = m / g
            l = l / g
        f[0], f[1] = (float(m), float(l))
    self.update_filling_sliders()
    self.push_fillings_to_manifold()