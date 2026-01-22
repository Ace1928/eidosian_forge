import collections
from .objects import ZERO_SHA, format_timezone, parse_timezone
def format_reflog_line(old_sha, new_sha, committer, timestamp, timezone, message):
    """Generate a single reflog line.

    Args:
      old_sha: Old Commit SHA
      new_sha: New Commit SHA
      committer: Committer name and e-mail
      timestamp: Timestamp
      timezone: Timezone
      message: Message
    """
    if old_sha is None:
        old_sha = ZERO_SHA
    return old_sha + b' ' + new_sha + b' ' + committer + b' ' + str(int(timestamp)).encode('ascii') + b' ' + format_timezone(timezone) + b'\t' + message