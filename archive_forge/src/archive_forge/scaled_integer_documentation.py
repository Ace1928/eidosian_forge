from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import six
Parses and returns a Binary scaled integer from string.

  All ISO/IEC prefixes are powers of 2: 1k == 1ki == 1024. This is a
  concession to the inconsistent mix of binary/decimal unit measures for
  memory capacity, disk capacity, cpu speed. Ideally ParseInteger should be
  used.

  Args:
    string: The string to parse the integer + units.
    default_unit: The default unit prefix name.
    type_abbr: The optional type abbreviation suffix, validated but otherwise
      ignored.

  Returns:
    The scaled integer value.
  