import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_direct_pass(self):
    self.model.attached_view = jv.JinjaView(text=self.TEMPL_STR)
    self.assertEqual('int is 1, string is value', str(self.model))