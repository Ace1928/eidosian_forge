from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml_location_value
from googlecloudsdk.core.util import files
from ruamel import yaml
import six
def strip_locations(obj):
    if list_like(obj):
        return [strip_locations(item) for item in obj]
    if dict_like(obj):
        return {key: strip_locations(value) for key, value in six.iteritems(obj)}
    return obj.value