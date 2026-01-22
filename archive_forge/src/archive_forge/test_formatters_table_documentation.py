import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
Get table output as a string, formatted according to
    CLI arguments, environment variables and terminal size

    tags - tuple of strings for data tags (column headers or fields)
    data - tuple of strings for single data row
         - list of tuples of strings for multiple rows of data
    extra_args - an instance of class args
               - a list of strings for CLI arguments
    