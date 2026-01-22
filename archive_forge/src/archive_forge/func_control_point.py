from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def control_point(self, gate_idx, wire_idx):
    """Draw a control point."""
    x = self._gate_grid[gate_idx]
    y = self._wire_grid[wire_idx]
    radius = self.control_radius
    c = Circle((x, y), radius * self.scale, ec='k', fc='k', fill=True, lw=self.linewidth)
    self._axes.add_patch(c)