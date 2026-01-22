from collections.abc import Iterable
import warnings
from typing import Sequence
def classical_wire(self, layers, wires) -> None:
    """Draw a classical control line.

        Args:
            layers: a list of x coordinates for the classical wire
            wires: a list of y coordinates for the classical wire. Wire numbers
                greater than the number of quantum wires will be scaled as classical wires.

        """
    outer_stroke = path_effects.Stroke(linewidth=5 * plt.rcParams['lines.linewidth'], foreground=plt.rcParams['lines.color'])
    inner_stroke = path_effects.Stroke(linewidth=3 * plt.rcParams['lines.linewidth'], foreground=plt.rcParams['figure.facecolor'])
    line = plt.Line2D(layers, [self._y(w) for w in wires], path_effects=[outer_stroke, inner_stroke], zorder=1)
    self.ax.add_line(line)