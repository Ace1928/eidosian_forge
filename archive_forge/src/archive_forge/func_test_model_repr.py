from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
def test_model_repr(self):
    model1 = base_model.ReportModel(data={'a': 1, 'b': 2}, attached_view=BasicView())
    model2 = base_model.ReportModel(data={'a': 1, 'b': 2})
    base_re = '<Model [^ ]+\\.[^ ]+ \\{.+\\} with '
    with_view_re = base_re + 'view [^ ]+\\.[^ ]+>'
    without_view_re = base_re + 'no view>'
    self.assertTrue(re.match(with_view_re, repr(model1)))
    self.assertTrue(re.match(without_view_re, repr(model2)))