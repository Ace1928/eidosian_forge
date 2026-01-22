import collections
import os
import re
import site
import traceback
from tensorflow.python.util import tf_stack
def parse_message(message):
    """Extract function tags and node tags from a message.

  Tags are named tuples representing the string {{type name}}. For example,
  in "123{{node Foo}}456{{function_node Bar}}789", there are two tags: a node
  tag and a function tag.

  Args:
    message: An error message, possibly from an OpError.

  Returns:
    A tuple containing the original message with function nodes stripped,
    function tags, and node tags.

    For example, if message is "123{{node Foo}}456{{function_node Bar}}789"
    then this function returns ("123{{node Foo}}456789",
    [_ParseTag("function_node", "Bar")], [_ParseTag("node", "Foo")]).
  """
    error_message = []
    func_tags = []
    node_tags = []
    pos = 0
    for match in re.finditer(_INTERPOLATION_PATTERN, message):
        parsed_tag = _ParseTag(match.group('type'), match.group('name'))
        if parsed_tag.type == 'function_node':
            error_message.append(match.group('sep'))
            func_tags.append(parsed_tag)
        else:
            error_message.append(match.group())
            node_tags.append(parsed_tag)
        pos = match.end()
    error_message.append(message[pos:])
    return (''.join(error_message), func_tags, node_tags)