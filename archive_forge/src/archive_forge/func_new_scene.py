from .gui import *
from .CyOpenGL import (HoroballScene, OpenGLOrthoWidget,
from plink.ipython_tools import IPythonTkRoot
import os
import sys
def new_scene(self, new_nbhd):
    self.nbhd = new_nbhd
    self.empty = self.nbhd is None
    self.set_ties()
    if new_nbhd and self.which_cusp >= new_nbhd.num_cusps():
        self.which_cusp = 0
    while self.volume_labels:
        label = self.volume_labels.pop()
        label.grid_forget()
        label.destroy()
    while self.cusp_sliders:
        slider = self.cusp_sliders.pop()
        slider.destroy()
    while self.slider_frames:
        frame = self.slider_frames.pop()
        frame.grid_forget()
        frame.destroy()
    while self.tie_buttons:
        button = self.tie_buttons.pop()
        button.grid_forget()
        button.destroy()
    while self.eye_buttons:
        button = self.eye_buttons.pop()
        button.grid_forget()
        button.destroy()
    self.eye_var.set(self.which_cusp)
    self.build_sliders()
    self.widget.tk.call(self.widget._w, 'makecurrent')
    self.scene = HoroballScene(new_nbhd, self.pgram_var, self.Ford_var, self.tri_var, self.horo_var, self.label_var, flipped=self.flip_var.get(), cutoff=self.cutoff, which_cusp=self.which_cusp, togl_widget=self.widget)
    assert self.scene is not None
    self.widget.redraw_impl = self.scene.draw
    self.configure_sliders()
    self.rebuild()