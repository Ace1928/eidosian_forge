import ast
from tempest.lib.cli import output_parser
import testtools
from manilaclient import api_versions
from manilaclient import config
def is_empty_row(row):
    empty_cells = 0
    for cell in row:
        if cell == '':
            empty_cells += 1
    return len(row) == empty_cells