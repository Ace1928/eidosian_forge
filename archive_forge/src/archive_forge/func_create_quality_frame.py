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
def create_quality_frame(self, parent):
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=0)
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=0)
    row = 0
    UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='maxSteps', title='Max Steps', row=row, left_end=1, right_end=100, update_function=self.widget.redraw_if_initialized)
    row += 1
    UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='maxDist', title='Max Distance', row=row, left_end=1.0, right_end=28.0, update_function=self.widget.redraw_if_initialized)
    row += 1
    UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='subpixelCount', title='Subpixel count', row=row, left_end=1.0, right_end=4.25, update_function=self.widget.redraw_if_initialized)
    return frame