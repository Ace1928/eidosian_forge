from pytest import warns, raises
from ase import Atoms
from ase import neb
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator

This is testing NEB in general, though at the moment focusing on the shared
calculator implementation that is replacing
the SingleCalculatorNEB class.
Intending to be a *true* unittest, by testing small things
