from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
import six
@property
def marshalled_dependencies(self):
    """Returns the marshalled dependencies or None if not marshalled."""
    return self._marshalled_dependencies