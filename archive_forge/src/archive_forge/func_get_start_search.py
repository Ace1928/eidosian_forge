from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import shlex
def get_start_search(self, namespace_name, prev_stop=0):
    """Gets the position to begin searching for an argument's value.

    This is a helper function for all the parse functions aside from parse().

    Args:
      namespace_name: The value of the argument in namespace's specified_args
      dictionary. ('INSTANCE_NAMES:1', '--zone', etc)
      prev_stop: Optional. For recurring flags, where the flag was last
      seen.

    Returns:
      The index in the example string to begin searching.
    """
    if prev_stop:
        start_search = self.example_string.find(namespace_name, prev_stop) + len(namespace_name)
    else:
        start_search = self.example_string.find(namespace_name) + len(namespace_name)
    if start_search == -1:
        command_name = self.command_node.name.replace('_', '-')
        command_name_start = self.example_string.find(command_name)
        start_search = command_name_start + len(command_name) + 1
    return start_search