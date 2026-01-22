from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
@property
def ready_condition(self):
    assert hasattr(self, 'READY_CONDITION')
    if self.conditions and self.READY_CONDITION in self.conditions:
        return self.conditions[self.READY_CONDITION]