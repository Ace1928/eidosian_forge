import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
def parse_csv_data_into_rows(self, csv_data, dialect, source):
    csv_reader = csv.reader([self.encode_for_csv(line + '\n') for line in csv_data], dialect=dialect)
    rows = []
    max_cols = 0
    for row in csv_reader:
        row_data = []
        for cell in row:
            cell_text = self.decode_from_csv(cell)
            cell_data = (0, 0, 0, statemachine.StringList(cell_text.splitlines(), source=source))
            row_data.append(cell_data)
        rows.append(row_data)
        max_cols = max(max_cols, len(row))
    return (rows, max_cols)