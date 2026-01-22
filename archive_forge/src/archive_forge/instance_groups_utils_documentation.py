from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
from six.moves import range  # pylint: disable=redefined-builtin
Add information if instance group is managed.

  Args:
    compute_holder: ComputeApiHolder, The compute API holder
    items: list of instance group messages,
    filter_mode: InstanceGroupFilteringMode, managed/unmanaged filtering options
  Returns:
    list of instance groups with computed dynamic properties
  