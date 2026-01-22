from __future__ import annotations
import os
from pymatgen.io.vasp import Potcar
def generate_potcar(args):
    """Generate POTCAR.

    Args:
        args (dict): Args from argparse.
    """
    if args.recursive:
        proc_dir(args.recursive, gen_potcar)
    elif args.symbols:
        try:
            p = Potcar(args.symbols, functional=args.functional)
            p.write_file('POTCAR')
        except Exception as exc:
            print(f'An error has occurred: {exc}')
    else:
        print('No valid options selected.')