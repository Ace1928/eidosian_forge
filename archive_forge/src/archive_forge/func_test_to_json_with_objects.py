import collections
import datetime
from lxml import etree
from oslo_serialization import jsonutils as json
import webob
from heat.common import serializers
from heat.tests import common
def test_to_json_with_objects(self):
    fixture = collections.OrderedDict([('is_public', True), ('value', complex(1, 2))])
    expected = '{"is_public": true, "value": "(1+2j)"}'
    actual = serializers.JSONResponseSerializer().to_json(fixture)
    self.assertEqual(json.loads(expected), json.loads(actual))