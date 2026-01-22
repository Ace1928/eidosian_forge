import ast
from tempest.lib.cli import output_parser
import testtools
from manilaclient import api_versions
from manilaclient import config
def is_embedded_table(parsed_rows):

    def is_table_border(t):
        return str(t).startswith('+')
    return isinstance(parsed_rows, list) and len(parsed_rows) > 3 and is_table_border(parsed_rows[0]) and is_table_border(parsed_rows[-1])