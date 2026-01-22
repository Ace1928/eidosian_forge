import io
import re
import pathlib
import numpy as np
from ase.calculators.lammps import convert
def extract_section(raw_datafile_contents, section_header):
    contents_split_by_section = split_contents_by_section(raw_datafile_contents)
    section = None
    for ind, block in enumerate(contents_split_by_section):
        if block.startswith(section_header):
            section = contents_split_by_section[ind + 1].strip()
            break
    return section