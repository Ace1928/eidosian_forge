import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def is_planar_isotopic(self, other, root=None, over_or_under=False) -> bool:
    return self.isosig() == other.isosig()