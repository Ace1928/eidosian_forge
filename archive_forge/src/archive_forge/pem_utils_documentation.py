from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
Formats a pem chain for output with exactly 1 newline character between each cert.

  Args:
    pem_chain: The list of certificate strings to output

  Returns:
    The string value of all certificates appended together for output.
  