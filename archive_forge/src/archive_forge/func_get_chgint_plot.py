from __future__ import annotations
import matplotlib.pyplot as plt
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.io.vasp import Chgcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.plotting import pretty_plot
def get_chgint_plot(args, ax: plt.Axes=None) -> plt.Axes:
    """Plot integrated charge.

    Args:
        args (dict): args from argparse.
        ax (plt.Axes): Matplotlib Axes object for plotting.

    Returns:
        plt.Axes: Matplotlib Axes object.
    """
    chgcar = Chgcar.from_file(args.chgcar_file)
    struct = chgcar.structure
    if args.inds:
        atom_ind = [int(i) for i in args.inds[0].split(',')]
    else:
        finder = SpacegroupAnalyzer(struct, symprec=0.1)
        sites = [sites[0] for sites in finder.get_symmetrized_structure().equivalent_sites]
        atom_ind = [struct.index(site) for site in sites]
    ax = ax or pretty_plot(12, 8)
    for idx in atom_ind:
        d = chgcar.get_integrated_diff(idx, args.radius, 30)
        ax.plot(d[:, 0], d[:, 1], label=f'Atom {idx} - {struct[idx].species_string}')
    ax.legend(loc='upper left')
    ax.set_xlabel('Radius (A)')
    ax.set_ylabel('Integrated charge (e)')
    plt.tight_layout()
    return ax