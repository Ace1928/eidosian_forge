from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
def remove_resource(self, resource_name, must_exist=True):
    if must_exist and resource_name not in self._api_data:
        raise ResourceNotFoundError(resource_name)
    del self._api_data[resource_name]