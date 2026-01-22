from .mcomplex_base import *
from .t3mlite import simplex

        Expects the result of
        Manifold._get_cusp_indices_and_peripheral_curve_data().

        It rearranges the Vertices of the mcomplex to match the ordering
        of the cusps in the SnapPea kernel and adds the peripheral curves
        in a format analogous to the kernel.
        