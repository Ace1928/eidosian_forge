from __future__ import annotations
import matplotlib.pyplot as plt
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.io.vasp import Chgcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.plotting import pretty_plot
def get_xrd_plot(args):
    """Plot XRD.

    Args:
        args (dict): Args from argparse
    """
    struct = Structure.from_file(args.xrd_structure_file)
    c = XRDCalculator()
    return c.get_plot(struct)