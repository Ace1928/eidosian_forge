import stat
from ... import controldir
def escape_commit_message(message):
    """Replace xml-incompatible control characters."""
    import re
    message, _ = re.subn('[^\t\n\r -\ud7ff\ue000-�]+', lambda match: match.group(0).encode('unicode_escape'), message)
    return message