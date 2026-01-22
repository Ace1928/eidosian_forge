from kivy import kivy_data_dir
from kivy.vector import Vector
from kivy.config import Config
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, \
from kivy.logger import Logger
from kivy.graphics import Color, BorderImage, Canvas
from kivy.core.image import Image
from kivy.resources import resource_find
from kivy.clock import Clock
from io import open
from os.path import join, splitext, basename
from os import listdir
from json import loads
def refresh_keys(self):
    layout = self.available_layouts[self.layout]
    layout_rows = layout['rows']
    layout_geometry = self.layout_geometry
    w, h = self.size
    kmtop, kmright, kmbottom, kmleft = self.key_margin
    uw_hint, uh_hint = layout_geometry['U_HINT']
    for line_nb in range(1, layout_rows + 1):
        llg = layout_geometry['LINE_%d' % line_nb] = []
        llg_append = llg.append
        for key in layout_geometry['LINE_HINT_%d' % line_nb]:
            x_hint, y_hint = key[0]
            w_hint, h_hint = key[1]
            kx = x_hint * w
            ky = y_hint * h
            kw = w_hint * w
            kh = h_hint * h
            kx = int(kx + kmleft)
            ky = int(ky + kmbottom)
            kw = int(kw - kmleft - kmright)
            kh = int(kh - kmbottom - kmtop)
            pos = (kx, ky)
            size = (kw, kh)
            llg_append((pos, size))
    self.layout_geometry = layout_geometry
    self.draw_keys()