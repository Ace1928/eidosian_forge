from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.core import log
Print the warning in last response.

  Args:
    response: The last response of series api call
    _: Represents unused_args

  Returns:
    Nested response, normally should be the resource of a LRO.
  