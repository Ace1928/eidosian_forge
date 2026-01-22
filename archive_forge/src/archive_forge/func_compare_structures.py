from __future__ import annotations
from tabulate import tabulate
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def compare_structures(args):
    """Compare structures in files for similarity using structure matcher.

    Args:
        args (dict): Args from argparse.
    """
    filenames = args.filenames
    if len(filenames) < 2:
        raise SystemExit('You need more than one structure to compare!')
    try:
        structures = [Structure.from_file(fn) for fn in filenames]
    except Exception as exc:
        print('Error converting file. Are they in the right format?')
        raise SystemExit(exc)
    matcher = StructureMatcher() if args.group == 'species' else StructureMatcher(comparator=ElementComparator())
    for idx, grp in enumerate(matcher.group_structures(structures)):
        print(f'Group {idx}: ')
        for s in grp:
            print(f'- {filenames[structures.index(s)]} ({s.formula})')
        print()