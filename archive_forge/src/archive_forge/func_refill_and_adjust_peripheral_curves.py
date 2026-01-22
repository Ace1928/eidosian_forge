from .geodesic_info import GeodesicInfo
from .geometric_structure import Filling, FillingMatrix
from ..snap.t3mlite import Mcomplex, simplex
from typing import Tuple, Optional, Sequence
def refill_and_adjust_peripheral_curves(manifold, post_drill_infos: Sequence[CuspPostDrillInfo]) -> None:
    manifold.dehn_fill([info.filling for info in post_drill_infos])
    for info in post_drill_infos:
        if info.peripheral_matrix is not None:
            manifold.set_peripheral_curves(info.peripheral_matrix, which_cusp=info.index)