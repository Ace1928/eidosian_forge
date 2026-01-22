from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_opt(string: str) -> dict[str, list]:
    """
        Read opt section from string.

        Args:
            string (str): String

        Returns:
            dict[str, list]: Opt section
        """
    patterns = {'CONSTRAINT': '^\\s*CONSTRAINT', 'FIXED': '^\\s*FIXED', 'DUMMY': '^\\s*DUMMY', 'CONNECT': '^\\s*CONNECT'}
    opt_matches = read_pattern(string, patterns)
    opt_sections = list(opt_matches)
    opt = {}
    if 'CONSTRAINT' in opt_sections:
        c_header = '^\\s*CONSTRAINT\\n'
        c_row = '(\\w.*)\\n'
        c_footer = '^\\s*ENDCONSTRAINT\\n'
        c_table = read_table_pattern(string, header_pattern=c_header, row_pattern=c_row, footer_pattern=c_footer)
        opt['CONSTRAINT'] = [val[0] for val in c_table[0]]
    if 'FIXED' in opt_sections:
        f_header = '^\\s*FIXED\\n'
        f_row = '(\\w.*)\\n'
        f_footer = '^\\s*ENDFIXED\\n'
        f_table = read_table_pattern(string, header_pattern=f_header, row_pattern=f_row, footer_pattern=f_footer)
        opt['FIXED'] = [val[0] for val in f_table[0]]
    if 'DUMMY' in opt_sections:
        d_header = '^\\s*DUMMY\\n'
        d_row = '(\\w.*)\\n'
        d_footer = '^\\s*ENDDUMMY\\n'
        d_table = read_table_pattern(string, header_pattern=d_header, row_pattern=d_row, footer_pattern=d_footer)
        opt['DUMMY'] = [val[0] for val in d_table[0]]
    if 'CONNECT' in opt_sections:
        cc_header = '^\\s*CONNECT\\n'
        cc_row = '(\\w.*)\\n'
        cc_footer = '^\\s*ENDCONNECT\\n'
        cc_table = read_table_pattern(string, header_pattern=cc_header, row_pattern=cc_row, footer_pattern=cc_footer)
        opt['CONNECT'] = [val[0] for val in cc_table[0]]
    return opt