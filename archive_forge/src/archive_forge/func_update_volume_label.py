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
def update_volume_label(self):
    try:
        vol_text = '%.3f' % self.widget.manifold.volume()
    except ValueError:
        vol_text = '-'
    sol_type = self.widget.manifold.solution_type(enum=True)
    sol_text = _solution_type_text[sol_type]
    try:
        self.vol_label.configure(text='Vol: %s (%s)' % (vol_text, sol_text))
    except AttributeError:
        pass