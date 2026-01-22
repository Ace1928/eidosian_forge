from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
def test_str_throws_error_with_no_attached_view(self):
    model = base_model.ReportModel(data={'c': [1, 2, 3]})
    self.assertRaisesRegex(Exception, 'Cannot stringify model: no attached view', str, model)