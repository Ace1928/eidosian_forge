import argparse
import ast
import re
import sys
def parse_ranges(range_string):
    """Parse a string representing numerical range(s).

  Args:
    range_string: (str) A string representing a numerical range or a list of
      them. For example:
        "[-1.0,1.0]", "[-inf, 0]", "[[-inf, -1.0], [1.0, inf]]"

  Returns:
    (list of list of float) A list of numerical ranges parsed from the input
      string.

  Raises:
    ValueError: If the input doesn't represent a range or a list of ranges.
  """
    range_string = range_string.strip()
    if not range_string:
        return []
    if 'inf' in range_string:
        range_string = re.sub('inf', repr(sys.float_info.max), range_string)
    ranges = ast.literal_eval(range_string)
    if isinstance(ranges, list) and (not isinstance(ranges[0], list)):
        ranges = [ranges]
    for item in ranges:
        if len(item) != 2:
            raise ValueError('Incorrect number of elements in range')
        elif not isinstance(item[0], (int, float)):
            raise ValueError('Incorrect type in the 1st element of range: %s' % type(item[0]))
        elif not isinstance(item[1], (int, float)):
            raise ValueError('Incorrect type in the 2nd element of range: %s' % type(item[0]))
    return ranges