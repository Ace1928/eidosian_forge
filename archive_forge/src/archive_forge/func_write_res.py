import glob
import re
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
def write_res(filename, images, write_info=True, write_results=True, significant_figures=6):
    """
    Write output in SHELX (.res) format

    To write multiple images, include a % format string in filename,
    e.g. `file_%03d.res`.

    Optionally include contents of Atoms.info dictionary if `write_info`
    is True, and/or results from attached calculator if `write_results`
    is True (only energy results are supported).
    """
    if not isinstance(images, (list, tuple)):
        images = [images]
    if len(images) > 1 and '%' not in filename:
        raise RuntimeError('More than one Atoms provided but no %' + ' format string found in filename')
    for i, atoms in enumerate(images):
        fn = filename
        if '%' in filename:
            fn = filename % i
        res = Res(atoms)
        if write_results:
            calculator = atoms.calc
            if calculator is not None and isinstance(calculator, Calculator):
                energy = calculator.results.get('energy')
                if energy is not None:
                    res.energy = energy
        res.write_file(fn, write_info=write_info, significant_figures=significant_figures)