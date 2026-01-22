from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.calliope import arg_parsers
Interpret the arg value as an item from an allowed value list.

    Args:
      arg_value: str. The value of the user input argument.

    Returns:
      The value of the arg.

    Raises:
      arg_parsers.ArgumentTypeError.
        If the arg value is not one of the allowed values.
    