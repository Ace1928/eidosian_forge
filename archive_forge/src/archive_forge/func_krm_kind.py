from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.command_lib.util.resource_map.declarative import declarative_map
from googlecloudsdk.core import exceptions
@property
def krm_kind(self):
    return KrmKind(self._resource.krm_group, self._resource.krm_kind)