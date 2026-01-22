import math
import os
import logging
from pathlib import Path
import warnings
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import matplotlib as mpl
from matplotlib import _api, cbook, cm
from matplotlib import _image
from matplotlib._image import *
import matplotlib.artist as martist
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
def flush_images():
    if len(image_group) == 1:
        image_group[0].draw(renderer)
    elif len(image_group) > 1:
        data, l, b = composite_images(image_group, renderer, mag)
        if data.size != 0:
            gc = renderer.new_gc()
            gc.set_clip_rectangle(parent.bbox)
            gc.set_clip_path(parent.get_clip_path())
            renderer.draw_image(gc, round(l), round(b), data)
            gc.restore()
    del image_group[:]