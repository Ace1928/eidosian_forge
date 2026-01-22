from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
def test_str_returns_string_with_attached_view(self):
    model = base_model.ReportModel(data={'a': 1, 'b': 2}, attached_view=BasicView())
    self.assertEqual(str(model), 'a: 1;b: 2;')