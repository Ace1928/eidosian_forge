from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def control_line(self, gate_idx, min_wire, max_wire):
    """Draw a vertical control line."""
    xdata = (self._gate_grid[gate_idx], self._gate_grid[gate_idx])
    ydata = (self._wire_grid[min_wire], self._wire_grid[max_wire])
    line = Line2D(xdata, ydata, color='k', lw=self.linewidth)
    self._axes.add_line(line)