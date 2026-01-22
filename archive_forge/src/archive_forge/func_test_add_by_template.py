from unittest import mock
from oslotest import base
from vitrageclient import exceptions as exc
from vitrageclient.tests.utils import get_resources_dir
from vitrageclient.v1.template import Template
def test_add_by_template(self):
    template = Template(mock.Mock())
    template.add(template_str=TEMPLATE_STRING)