import io
import re
import pathlib
import numpy as np
from ase.calculators.lammps import convert
def lammpsdata_file_extracted_sections(lammpsdata):
    """
    Manually read a lammpsdata file and grep for the different
    quantities we want to check.  Accepts either a string indicating the name
    of the file, a pathlib.Path object indicating the location of the file, a
    StringIO object containing the file contents, or a file object
    """
    if isinstance(lammpsdata, str) or isinstance(lammpsdata, pathlib.Path):
        with open(lammpsdata) as fd:
            raw_datafile_contents = fd.read()
    elif isinstance(lammpsdata, io.StringIO):
        raw_datafile_contents = lammpsdata.getvalue()
    elif isinstance(lammpsdata, io.TextIOBase):
        raw_datafile_contents = lammpsdata.read()
    else:
        raise ValueError('Lammps data file content inputted in unsupported object type {type(lammpsdata)}')
    cell = extract_cell(raw_datafile_contents)
    mass = extract_mass(raw_datafile_contents)
    charges, positions, travels = extract_atom_quantities(raw_datafile_contents)
    velocities = extract_velocities(raw_datafile_contents)
    return {'cell': cell, 'mass': mass, 'charges': charges, 'positions': positions, 'travels': travels, 'velocities': velocities}