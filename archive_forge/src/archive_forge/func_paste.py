import errno
import subprocess
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse
import requests
import unicodedata
from .config import getpreferredencoding
from .translations import _
from ._typing_compat import Protocol
def paste(self, s: str) -> Tuple[str, None]:
    """Call out to helper program for pastebin upload."""
    try:
        helper = subprocess.Popen('', executable=self.executable, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        assert helper.stdin is not None
        encoding = getpreferredencoding()
        helper.stdin.write(s.encode(encoding))
        output = helper.communicate()[0].decode(encoding)
        paste_url = output.split()[0]
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise PasteFailed(_('Helper program not found.'))
        else:
            raise PasteFailed(_('Helper program could not be run.'))
    if helper.returncode != 0:
        raise PasteFailed(_('Helper program returned non-zero exit status %d.' % (helper.returncode,)))
    if not paste_url:
        raise PasteFailed(_('No output from helper program.'))
    parsed_url = urlparse(paste_url)
    if not parsed_url.scheme or any((unicodedata.category(c) == 'Cc' for c in paste_url)):
        raise PasteFailed(_("Failed to recognize the helper program's output as an URL."))
    return (paste_url, None)