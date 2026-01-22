from io import StringIO
import tempfile
from typing import IO
from typing import Union
from _pytest.config import Config
from _pytest.config import create_terminal_writer
from _pytest.config.argparsing import Parser
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
import pytest
def create_new_paste(contents: Union[str, bytes]) -> str:
    """Create a new paste using the bpaste.net service.

    :contents: Paste contents string.
    :returns: URL to the pasted contents, or an error message.
    """
    import re
    from urllib.parse import urlencode
    from urllib.request import urlopen
    params = {'code': contents, 'lexer': 'text', 'expiry': '1week'}
    url = 'https://bpa.st'
    try:
        response: str = urlopen(url, data=urlencode(params).encode('ascii')).read().decode('utf-8')
    except OSError as exc_info:
        return 'bad response: %s' % exc_info
    m = re.search('href="/raw/(\\w+)"', response)
    if m:
        return f'{url}/show/{m.group(1)}'
    else:
        return "bad response: invalid format ('" + response + "')"