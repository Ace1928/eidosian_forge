from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def not_point(self, gate_idx, wire_idx):
    """Draw a NOT gates as the circle with plus in the middle."""
    x = self._gate_grid[gate_idx]
    y = self._wire_grid[wire_idx]
    radius = self.not_radius
    c = Circle((x, y), radius, ec='k', fc='w', fill=False, lw=self.linewidth)
    self._axes.add_patch(c)
    l = Line2D((x, x), (y - radius, y + radius), color='k', lw=self.linewidth)
    self._axes.add_line(l)