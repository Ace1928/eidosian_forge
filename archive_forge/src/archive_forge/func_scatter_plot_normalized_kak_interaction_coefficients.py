import cmath
import math
from typing import (
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq._import import LazyLoader
from cirq.linalg import combinators, diagonalize, predicates, transformations
def scatter_plot_normalized_kak_interaction_coefficients(interactions: Iterable[Union[np.ndarray, 'cirq.SupportsUnitary', 'KakDecomposition']], *, include_frame: bool=True, ax: Optional[mplot3d.axes3d.Axes3D]=None, **kwargs):
    """Plots the interaction coefficients of many two-qubit operations.

    Plots:
        A point for the (x, y, z) normalized interaction coefficients of
        each interaction from the given interactions. The (x, y, z) coordinates
        are normalized so that the maximum value is at 1 instead of at pi/4.

        If `include_frame` is set to True, then a black wireframe outline of the
        canonicalized normalized KAK coefficient space. The space is defined by
        the following two constraints:

            0 <= abs(z) <= y <= x <= 1
            if x = 1 then z >= 0

        The wireframe includes lines along the surface of the space at z=0.

        The space is a prism with the identity at the origin, a crease along
        y=z=0 leading to the CZ/CNOT at x=1 and a vertical triangular face that
        contains the iswap at x=y=1,z=0 and the swap at x=y=z=1:

                                 (x=1,y=1,z=0)
                             swap___iswap___swap (x=1,y=1,z=+-1)
                               _/\\    |    /
                             _/   \\   |   /
                           _/      \\  |  /
                         _/         \\ | /
                       _/            \\|/
        (x=0,y=0,z=0) I---------------CZ (x=1,y=0,z=0)

    Args:
        interactions: An iterable of two qubit unitary interactions. Each
            interaction can be specified as a raw 4x4 unitary matrix, or an
            object with a 4x4 unitary matrix according to `cirq.unitary` (
            (e.g. `cirq.CZ` or a `cirq.KakDecomposition` or a `cirq.Circuit`
            over two qubits).
        include_frame: Determines whether or not to draw the kak space
            wireframe. Defaults to `True`.
        ax: A matplotlib 3d axes object to plot into. If not specified, a new
            figure is created, plotted, and shown.

        **kwargs: Arguments forwarded into the call to `scatter` that plots the
            points. Working arguments include color `c='blue'`, scale `s=2`,
            labelling `label="theta=pi/4"`, etc. For reference see the
            `matplotlib.pyplot.scatter` documentation:
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html

    Returns:
        The matplotlib 3d axes object that was plotted into.

    Examples:
        >>> ax = None
        >>> for y in np.linspace(0, 0.5, 4):
        ...     a, b = cirq.LineQubit.range(2)
        ...     circuits = [
        ...         cirq.Circuit(
        ...             cirq.CZ(a, b)**0.5,
        ...             cirq.X(a)**y, cirq.X(b)**x,
        ...             cirq.CZ(a, b)**0.5,
        ...             cirq.X(a)**x, cirq.X(b)**y,
        ...             cirq.CZ(a, b) ** 0.5,
        ...         )
        ...         for x in np.linspace(0, 1, 25)
        ...     ]
        ...     ax = cirq.scatter_plot_normalized_kak_interaction_coefficients(
        ...         circuits,
        ...         include_frame=ax is None,
        ...         ax=ax,
        ...         s=1,
        ...         label=f'y={y:0.2f}')
        >>> _ = ax.legend()
        >>> import matplotlib.pyplot as plt
        >>> plt.show()
    """
    show_plot = not ax
    if not ax:
        fig = plt.figure()
        ax = cast(mplot3d.axes3d.Axes3D, fig.add_subplot(1, 1, 1, projection='3d'))

    def coord_transform(pts: Union[List[Tuple[int, int, int]], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(pts) == 0:
            return (np.array([]), np.array([]), np.array([]))
        xs, ys, zs = np.transpose(pts)
        return (xs, zs, ys)
    if include_frame:
        envelope = [(0, 0, 0), (1, 1, 1), (1, 1, -1), (0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 0, 0), (1, 1, -1), (1, 0, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0)]
        ax.plot(*coord_transform(envelope), c='black', linewidth=1)
    if not isinstance(interactions, np.ndarray):
        interactions_extracted: List[np.ndarray] = [a if isinstance(a, np.ndarray) else protocols.unitary(a) for a in interactions]
    else:
        interactions_extracted = [interactions]
    points = kak_vector(interactions_extracted) * 4 / np.pi
    ax.scatter(*coord_transform(points), **kwargs)
    ax.set_xlim(0, +1)
    ax.set_ylim(-1, +1)
    ax.set_zlim(0, +1)
    if show_plot:
        fig.show()
    return ax