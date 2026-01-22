import argparse
import ast
import re
import sys
def parse_indices(indices_string):
    """Parse a string representing indices.

  For example, if the input is "[1, 2, 3]", the return value will be a list of
  indices: [1, 2, 3]

  Args:
    indices_string: (str) a string representing indices. Can optionally be
      surrounded by a pair of brackets.

  Returns:
    (list of int): Parsed indices.
  """
    indices_string = re.sub('\\s+', '', indices_string)
    if indices_string.startswith('[') and indices_string.endswith(']'):
        indices_string = indices_string[1:-1]
    return [int(element) for element in indices_string.split(',')]