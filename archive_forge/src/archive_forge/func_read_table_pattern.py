from __future__ import annotations
import logging
import os
import re
import warnings
from glob import glob
from itertools import chain
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from monty.re import regrep
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.cp2k.inputs import Keyword
from pymatgen.io.cp2k.sets import Cp2kInput
from pymatgen.io.cp2k.utils import natural_keys, postprocessor
from pymatgen.io.xyz import XYZ
def read_table_pattern(self, header_pattern, row_pattern, footer_pattern, postprocess=str, attribute_name=None, last_one_only=True, strip=None):
    """
        This function originally comes from pymatgen.io.vasp.outputs Outcar class.

        Parse table-like data. A table composes of three parts: header,
        main body, footer. All the data matches "row pattern" in the main body
        will be returned.

        Args:
            header_pattern (str): The regular expression pattern matches the
                table header. This pattern should match all the text
                immediately before the main body of the table. For multiple
                sections table match the text until the section of
                interest. MULTILINE and DOTALL options are enforced, as a
                result, the "." meta-character will also match "\\n" in this
                section.
            row_pattern (str): The regular expression matches a single line in
                the table. Capture interested field using regular expression
                groups.
            footer_pattern (str): The regular expression matches the end of the
                table. E.g. a long dash line.
            postprocess (callable): A post processing function to convert all
                matches. Defaults to str, i.e., no change.
            attribute_name (str): Name of this table. If present the parsed data
                will be attached to "data. e.g. self.data["efg"] = [...]
            last_one_only (bool): All the tables will be parsed, if this option
                is set to True, only the last table will be returned. The
                enclosing list will be removed. i.e. Only a single table will
                be returned. Default to be True.
            strip (list): Whether or not to strip contents out of the file before
                reading for a table pattern. This is mainly used by parse_scf_opt(),
                to strip HFX info out of the SCF loop start or DFT+U warnings out
                of the SCF loop iterations.

        Returns:
            List of tables. 1) A table is a list of rows. 2) A row if either a list of
            attribute values in case the the capturing group is defined without name in
            row_pattern, or a dict in case that named capturing groups are defined by
            row_pattern.
        """
    with zopen(self.filename, mode='rt') as file:
        if strip:
            lines = file.readlines()
            text = ''.join([lines[i] for i in range(1, len(lines) - 1) if all((not lines[i].strip().startswith(c) and (not lines[i - 1].strip().startswith(c)) and (not lines[i + 1].strip().startswith(c)) for c in strip))])
        else:
            text = file.read()
    table_pattern_text = header_pattern + '\\s*^(?P<table_body>(?:\\s+' + row_pattern + ')+)\\s+' + footer_pattern
    table_pattern = re.compile(table_pattern_text, re.MULTILINE | re.DOTALL)
    rp = re.compile(row_pattern)
    tables = []
    for mt in table_pattern.finditer(text):
        table_body_text = mt.group('table_body')
        table_contents = []
        for line in table_body_text.split('\n'):
            ml = rp.search(line)
            if ml is None:
                continue
            d = ml.groupdict()
            if len(d) > 0:
                processed_line = {k: postprocess(v) for k, v in d.items()}
            else:
                processed_line = [postprocess(v) for v in ml.groups()]
            table_contents.append(processed_line)
        tables.append(table_contents)
    retained_data = tables[-1] if last_one_only else tables
    if attribute_name is not None:
        self.data[attribute_name] = retained_data
    return retained_data