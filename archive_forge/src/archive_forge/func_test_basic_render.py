from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
def test_basic_render(self):
    self.report.add_section(BasicView(), basic_generator)
    self.assertEqual(self.report.run(), 'int: 1;string: value;')