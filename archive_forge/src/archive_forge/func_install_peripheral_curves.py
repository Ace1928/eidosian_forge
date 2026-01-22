from ..snap.t3mlite import Mcomplex
from ..snap.t3mlite import simplex, Tetrahedron
from collections import deque
from typing import Dict
def install_peripheral_curves(start_tet: Tetrahedron) -> None:
    """
    Given a suitable base tetrahedron in the complex obtained by
    crushing edges in the barycentric subdivision, compute a new
    meridian and longitude.
    """
    _install_meridian(start_tet)
    _install_longitude(start_tet)