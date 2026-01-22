import argparse
import ast
import re
import sys
def parse_time_interval(interval_str):
    """Convert a human-readable time interval to a tuple of start and end value.

  Args:
    interval_str: (`str`) A human-readable str representing an interval
      (e.g., "[10us, 20us]", "<100s", ">100ms"). Supported time suffixes are
      us, ms, s.

  Returns:
    `Interval` object where start and end are in microseconds.

  Raises:
    ValueError: if the input is not valid.
  """
    str_interval = _parse_interval(interval_str)
    interval_start = 0
    interval_end = float('inf')
    if str_interval.start:
        interval_start = parse_readable_time_str(str_interval.start)
    if str_interval.end:
        interval_end = parse_readable_time_str(str_interval.end)
    if interval_start > interval_end:
        raise ValueError('Invalid interval %s. Start must be before end of interval.' % interval_str)
    return Interval(interval_start, str_interval.start_included, interval_end, str_interval.end_included)