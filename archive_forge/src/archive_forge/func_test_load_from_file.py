import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
@mock.mock_open(MM_OPEN)
def test_load_from_file(self):
    self.model.attached_view = jv.JinjaView(path='a/b/c/d.jinja.txt')
    self.assertEqual('int is 1, string is value', str(self.model))
    self.MM_FILE.assert_called_with_once('a/b/c/d.jinja.txt')