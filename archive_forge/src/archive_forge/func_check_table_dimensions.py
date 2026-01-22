import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
def check_table_dimensions(self, rows, header_rows, stub_columns):
    if len(rows) < header_rows:
        error = self.state_machine.reporter.error('%s header row(s) specified but only %s row(s) of data supplied ("%s" directive).' % (header_rows, len(rows), self.name), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
        raise SystemMessagePropagation(error)
    if len(rows) == header_rows > 0:
        error = self.state_machine.reporter.error('Insufficient data supplied (%s row(s)); no data remaining for table body, required by "%s" directive.' % (len(rows), self.name), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
        raise SystemMessagePropagation(error)
    for row in rows:
        if len(row) < stub_columns:
            error = self.state_machine.reporter.error('%s stub column(s) specified but only %s columns(s) of data supplied ("%s" directive).' % (stub_columns, len(row), self.name), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            raise SystemMessagePropagation(error)
        if len(row) == stub_columns > 0:
            error = self.state_machine.reporter.error('Insufficient data supplied (%s columns(s)); no data remaining for table body, required by "%s" directive.' % (len(row), self.name), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            raise SystemMessagePropagation(error)