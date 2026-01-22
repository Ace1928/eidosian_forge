import ast
from tempest.lib.cli import output_parser
import testtools
from manilaclient import api_versions
from manilaclient import config
def get_column_index(column_name, headers, default):
    return next((i for i, h in enumerate(headers) if h.lower() == column_name), default)