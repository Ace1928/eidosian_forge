from .mcomplex_base import *
from .t3mlite import simplex
def reindex_cusps_and_transfer_peripheral_curves(self):
    self.reindex_cusps_and_add_peripheral_curves(self.snappyTriangulation._get_cusp_indices_and_peripheral_curve_data())