import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_recursive_type_propogation_on_recursive_structures(self):
    nested_model = mwdv_generator()
    data = {'a': 1, 'b': [nested_model]}
    nested_model['c'] = data
    top_model = base_model.ReportModel(data=data)
    top_model.set_current_view_type('json')
    self.assertEqual(nested_model.attached_view, nested_model.views['json'])
    del nested_model['c']