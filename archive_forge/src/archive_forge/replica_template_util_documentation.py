from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
Apply the params provided into the template.

  Args:
    node: A node in the parsed template
    params: a dict of params of param-name -> param-value

  Returns:
    A tuple of (new_node, missing_params, used_params) where new_node is the
    node with all params replaced, missing_params is set of param
    references found in the node that were not provided and used_params were
    the params that we actually used.
  