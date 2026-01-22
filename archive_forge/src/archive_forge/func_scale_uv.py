import math
from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def scale_uv(self):
    """
        Scales u and v to avoid overlap of the arrows.

        u and v are added to x and y to get the
        endpoints of the arrows so a smaller scale value will
        result in less overlap of arrows.
        """
    self.u = [i * self.scale * self.scaleratio for i in self.u]
    self.v = [i * self.scale for i in self.v]