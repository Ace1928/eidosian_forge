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
def create_fillings_frame(self, parent):
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    row = 0
    self.filling_controllers = []
    for i in range(self.widget.manifold.num_cusps()):
        scale_m = ZoomSlider(frame, left_end=-15.0, right_end=15.0, label_text='Cusp %d' % i, on_change=self.focus_viewer)
        scale_m.grid(row=row, column=0, sticky=tkinter.NSEW)
        self.filling_controllers.append(UniformDictController(self.filling_dict, key='fillings', index=i, component_index=0, update_function=self.push_fillings_to_manifold, scale=scale_m))
        scale_l = ZoomSlider(frame, left_end=-15.0, right_end=15.0, on_change=self.focus_viewer)
        scale_l.grid(row=row, column=1, sticky=tkinter.NSEW)
        self.filling_controllers.append(UniformDictController(self.filling_dict, key='fillings', index=i, component_index=1, update_function=self.push_fillings_to_manifold, scale=scale_l))
        row += 1
    frame.rowconfigure(row, weight=1)
    subframe = ttk.Frame(frame)
    subframe.grid(row=row, column=0, columnspan=5)
    subframe.columnconfigure(0, weight=1)
    subframe.columnconfigure(1, weight=0)
    subframe.columnconfigure(2, weight=0)
    subframe.columnconfigure(3, weight=0)
    subframe.columnconfigure(4, weight=1)
    recompute_button = ttk.Button(subframe, text='Recompute hyp. structure', takefocus=0, command=self.recompute_hyperbolic_structure)
    recompute_button.grid(row=0, column=1)
    orb_button = ttk.Button(subframe, text='Make orbifold', takefocus=0, command=self.make_orbifold)
    orb_button.grid(row=0, column=2)
    mfd_button = ttk.Button(subframe, text='Make manifold', takefocus=0, command=self.make_manifold)
    mfd_button.grid(row=0, column=3)
    return frame