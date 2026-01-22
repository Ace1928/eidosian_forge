import tkinter
import math
import sys
from tkinter import ttk
from .gui_utilities import UniformDictController, FpsLabelUpdater
from .raytracing_view import *
from .hyperboloid_utilities import unit_3_vector_and_distance_to_O13_hyperbolic_translation
from .zoom_slider import Slider, ZoomSlider
def round_fillings(self):
    for f in self.filling_dict['fillings'][1]:
        for i in [0, 1]:
            f[i] = float(round(f[i]))
    self.update_filling_sliders()
    self.push_fillings_to_manifold()