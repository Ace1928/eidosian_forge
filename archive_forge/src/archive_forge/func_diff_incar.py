from __future__ import annotations
import argparse
import itertools
from tabulate import tabulate, tabulate_formats
from pymatgen.cli.pmg_analyze import analyze
from pymatgen.cli.pmg_config import configure_pmg
from pymatgen.cli.pmg_plot import plot
from pymatgen.cli.pmg_potcar import generate_potcar
from pymatgen.cli.pmg_structure import analyze_structures
from pymatgen.core import SETTINGS
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Incar, Potcar
def diff_incar(args):
    """Handle diff commands.

    Args:
        args: Args from command.
    """
    filepath1 = args.incars[0]
    filepath2 = args.incars[1]
    incar1 = Incar.from_file(filepath1)
    incar2 = Incar.from_file(filepath2)

    def format_lists(v):
        if isinstance(v, (tuple, list)):
            return ' '.join((f'{len(tuple(group))}*{i:.2f}' for i, group in itertools.groupby(v)))
        return v
    diff = incar1.diff(incar2)
    output = [['SAME PARAMS', '', ''], ['---------------', '', ''], ['', '', ''], ['DIFFERENT PARAMS', '', ''], ['----------------', '', '']]
    output += [(k, format_lists(diff['Same'][k]), format_lists(diff['Same'][k])) for k in sorted(diff['Same']) if k != 'SYSTEM']
    output += [(k, format_lists(diff['Different'][k]['INCAR1']), format_lists(diff['Different'][k]['INCAR2'])) for k in sorted(diff['Different']) if k != 'SYSTEM']
    print(tabulate(output, headers=['', filepath1, filepath2]))
    return 0