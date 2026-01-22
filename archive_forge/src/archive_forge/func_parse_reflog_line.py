import collections
from .objects import ZERO_SHA, format_timezone, parse_timezone
def parse_reflog_line(line):
    """Parse a reflog line.

    Args:
      line: Line to parse
    Returns: Tuple of (old_sha, new_sha, committer, timestamp, timezone,
        message)
    """
    begin, message = line.split(b'\t', 1)
    old_sha, new_sha, rest = begin.split(b' ', 2)
    committer, timestamp_str, timezone_str = rest.rsplit(b' ', 2)
    return Entry(old_sha, new_sha, committer, int(timestamp_str), parse_timezone(timezone_str)[0], message)