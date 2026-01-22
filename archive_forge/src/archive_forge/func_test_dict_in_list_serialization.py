import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_dict_in_list_serialization(self):
    self.model['lt'] = [1, {'b': 2, 'c': 3}]
    target_str = 'int = 1\nlt = \n  1\n  [dict]\n    b = 2\n    c = 3\nstring = value'
    self.assertEqual(target_str, str(self.model))