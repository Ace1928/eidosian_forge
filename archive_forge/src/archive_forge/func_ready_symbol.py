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
def ready_symbol(self):
    """Return a symbol summarizing the status of this object."""
    return self.ReadySymbolAndColor()[0]