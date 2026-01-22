from ...sage_helper import _within_sage
from ...math_basics import correct_max
from ...snap.kernel_structures import *
from ...snap.fundamental_polyhedron import *
from ...snap.mcomplex_base import *
from ...snap.t3mlite import simplex
from ...snap import t3mlite as t3m
from ...exceptions import InsufficientPrecisionError
from ..cuspCrossSection import ComplexCuspCrossSection
from ..upper_halfspace.ideal_point import *
from ..interval_tree import *
from .cusp_translate_engine import *
import heapq
def record_unglued_generator(self, tile, g):
    heapq.heappush(self.unglued_generator_heapq, CuspTilingEngine.UngluedGenerator(tile=tile, g=g, height_upper_bound=self.upper_bound_for_height_of_unglued_generator(tile, g)))