import collections
import warnings
from sympy.external import import_module
def declare_phy_entities(self, ctx, phy_type, i, j=None):
    if phy_type in ('frame', 'newtonian'):
        declare_frames(self, ctx, i, j)
    elif phy_type == 'particle':
        declare_particles(self, ctx, i, j)
    elif phy_type == 'point':
        declare_points(self, ctx, i, j)
    elif phy_type == 'bodies':
        declare_bodies(self, ctx, i, j)