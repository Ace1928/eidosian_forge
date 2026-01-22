from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
@property
def yaml(self):
    if len(self._data) == 1:
        return str(self._data[0])
    return '---\n'.join([str(x) for x in self._data])