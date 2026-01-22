from typing import List, Union
from functools import reduce
import colorsys
import numpy as np
from qiskit import user_config
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import PauliList, SparsePauliOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.utils import optionals as _optionals
from qiskit.circuit.tools.pi_check import pi_check
from .array import _num_to_latex, array_to_latex
from .utils import matplotlib_close_if_inline
from .exceptions import VisualizationError
@_optionals.HAS_MATPLOTLIB.require_in_call
def generate_facecolors(x, y, z, dx, dy, dz, color):
    """Generates shaded facecolors for shaded bars.

    This is here to work around a Matplotlib bug
    where alpha does not work in Bar3D.

    Args:
        x (array_like): The x- coordinates of the anchor point of the bars.
        y (array_like): The y- coordinates of the anchor point of the bars.
        z (array_like): The z- coordinates of the anchor point of the bars.
        dx (array_like): Width of bars.
        dy (array_like): Depth of bars.
        dz (array_like): Height of bars.
        color (array_like): sequence of valid color specifications, optional
    Returns:
        list: Shaded colors for bars.
    Raises:
        MissingOptionalLibraryError: If matplotlib is not installed
    """
    import matplotlib.colors as mcolors
    cuboid = np.array([((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))])
    polys = np.empty(x.shape + cuboid.shape)
    for i, p, dp in [(0, x, dx), (1, y, dy), (2, z, dz)]:
        p = p[..., np.newaxis, np.newaxis]
        dp = dp[..., np.newaxis, np.newaxis]
        polys[..., i] = p + dp * cuboid[..., i]
    polys = polys.reshape((-1,) + polys.shape[2:])
    facecolors = []
    if len(color) == len(x):
        for c in color:
            facecolors.extend([c] * 6)
    else:
        facecolors = list(mcolors.to_rgba_array(color))
        if len(facecolors) < len(x):
            facecolors *= 6 * len(x)
    normals = _generate_normals(polys)
    return _shade_colors(facecolors, normals)