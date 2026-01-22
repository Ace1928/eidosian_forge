from __future__ import annotations
from tabulate import tabulate
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def convert_fmt(args):
    """Convert files from one format to another.

    Args:
        args (dict): Args from argparse.
    """
    if len(args.filenames) != 2:
        print('File format conversion takes in only two filenames.')
    struct = Structure.from_file(args.filenames[0], primitive='prim' in args.filenames[1].lower())
    struct.to(filename=args.filenames[1])