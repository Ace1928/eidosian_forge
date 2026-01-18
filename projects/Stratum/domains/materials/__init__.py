"""
Materials domain for Stratum.

This subpackage contains classes and helpers that model the high
energy (physics) aspects of the simulation such as the equation of
state, fusion/decay, degeneracy and black hole formation. The core
class exposed is ``MaterialsFundamentals`` which encapsulates the
behaviour of a small number of fundamental materials used in the
prototype: a generic ``StellarGas`` species, a degenerate matter
species, and whatever additional species emerge during fusion and
decay.
"""

from .fundamentals import MaterialsFundamentals

__all__ = ["MaterialsFundamentals"]