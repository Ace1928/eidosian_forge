from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import shlex
def parse_nested_list(self, list_arg, key, value):
    """Gets metadata from an example command string for nested list arguments.

    This is a helper function for parse_list().

    It updates the already existing ExampleCommandMetadata object of the example
    string with the metadata.

    Args:
      list_arg: The list-type argument to collect metadata about.
      key: The name of the argument's attribute in the example string's
      namespace.
      value: The actual input representing the flag/argument in the example
      string (e.g --zone, --shielded-secure-boot).
    """
    flag_count = self.example_string.count(value)
    if isinstance(list_arg[0], collections.OrderedDict):
        if flag_count == 1 and len(list_arg) > 1:
            first_dict = list(list_arg[0].items())
            last_dict = list(list_arg[-1].items())
            start = first_dict[0][0]
            stop = last_dict[-1][1]
            start_search = self.get_start_search(value)
            start_index = self.example_string.index(str(start), start_search)
            next_start = self.example_string.find('--', start_index)
            if next_start < start_search:
                next_start = len(self.example_string)
            stop_index = self.example_string.rfind(str(stop), start_search, next_start) + len(str(stop)) - 1
            arg_value = self.example_string[start_index:stop_index + 1]
            arg_metadata = ArgumentMetadata(key, arg_value, key, start_index, stop_index)
            self.example_metadata.add_arg_metadata(arg_metadata)
        else:
            prev_stop = 0
            for i in range(flag_count):
                val = list_arg[i]
                prev_stop = self.parse_dict(val, key, value, prev_stop, i + 1)
    else:
        prev_stop = 0
        for i in range(flag_count):
            val = list_arg[i]
            prev_stop = self.parse_list(val, key, value, prev_stop, i + 1)