import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def draw_pane(self, renderer):
    """
        Draw pane.

        Parameters
        ----------
        renderer : `~matplotlib.backend_bases.RendererBase` subclass
        """
    renderer.open_group('pane3d', gid=self.get_gid())
    xys, loc = self.active_pane(renderer)
    self.pane.xy = xys[:, :2]
    self.pane.draw(renderer)
    renderer.close_group('pane3d')