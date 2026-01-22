import argparse
import ast
import re
import sys
def parse_memory_interval(interval_str):
    """Convert a human-readable memory interval to a tuple of start and end value.

  Args:
    interval_str: (`str`) A human-readable str representing an interval
      (e.g., "[10kB, 20kB]", "<100M", ">100G"). Only the units "kB", "MB", "GB"
      are supported. The "B character at the end of the input `str` may be
      omitted.

  Returns:
    `Interval` object where start and end are in bytes.

  Raises:
    ValueError: if the input is not valid.
  """
    str_interval = _parse_interval(interval_str)
    interval_start = 0
    interval_end = float('inf')
    if str_interval.start:
        interval_start = parse_readable_size_str(str_interval.start)
    if str_interval.end:
        interval_end = parse_readable_size_str(str_interval.end)
    if interval_start > interval_end:
        raise ValueError('Invalid interval %s. Start of interval must be less than or equal to end of interval.' % interval_str)
    return Interval(interval_start, str_interval.start_included, interval_end, str_interval.end_included)