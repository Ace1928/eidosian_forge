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
def push_fillings_to_manifold(self):
    self.show_failed_dirichlet(show=False)
    self.widget.manifold.dehn_fill(self.filling_dict['fillings'][1])
    self.widget.recompute_raytracing_data_and_redraw()
    self.update_volume_label()
    self.reset_geodesics()
    if self.fillings_changed_callback:
        self.fillings_changed_callback()