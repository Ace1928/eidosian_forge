import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
def extend_short_rows_with_empty_cells(self, columns, parts):
    for part in parts:
        for row in part:
            if len(row) < columns:
                row.extend([(0, 0, 0, [])] * (columns - len(row)))