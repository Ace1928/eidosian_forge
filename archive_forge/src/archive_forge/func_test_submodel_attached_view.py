from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
def test_submodel_attached_view(self):

    class TmpView(object):

        def __call__(self, model):
            return '{len: ' + str(len(model.c)) + '}'

    def generate_model_with_submodel():
        base_m = basic_generator()
        tv = TmpView()
        base_m['submodel'] = base_model.ReportModel(data={'c': [1, 2, 3]}, attached_view=tv)
        return base_m
    self.assertEqual(BasicView()(generate_model_with_submodel()), 'int: 1;string: value;submodel: {len: 3};')