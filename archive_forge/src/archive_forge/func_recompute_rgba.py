from __future__ import absolute_import
import numpy as np
import ipywidgets as widgets  # we should not have widgets under two names
import traitlets
from traitlets import Unicode
from traittypes import Array
import ipyvolume._version
from ipyvolume import serialize
def recompute_rgba(self, *_ignore):
    import matplotlib
    rgba = np.zeros((1024, 4))
    N = range(1, 4)
    levels = [getattr(self, 'level%d' % k) for k in N]
    opacities = [getattr(self, 'opacity%d' % k) for k in N]
    widths = [getattr(self, 'width%d' % k) for k in N]
    colors = [np.array(matplotlib.colors.colorConverter.to_rgb(name)) for name in ['red', 'green', 'blue']]
    for i in range(rgba.shape[0]):
        position = i / 1023.0
        intensity = 0.0
        for j in range(3):
            intensity = np.exp(-((position - levels[j]) / widths[j]) ** 2)
            rgba[i, 0:3] += colors[j] * opacities[j] * intensity
            rgba[i, 3] += opacities[j] * intensity
        rgba[i, 0:3] /= rgba[i, 0:3].max()
    rgba = np.clip(rgba, 0, 1)
    self.rgba = rgba