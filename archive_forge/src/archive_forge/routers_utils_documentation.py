from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import operator
from six.moves import map  # pylint: disable=redefined-builtin
Parses a list of IP address ranges into CustomLearnedIpRange objects.

  Args:
    messages: API messages holder.
    ip_ranges: A list of ip_ranges, where each ip_range is a CIDR-formatted IP.

  Returns:
    A list of CustomLearnedIpRange objects containing the specified IP ranges.
  