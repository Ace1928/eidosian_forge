from __future__ import division
import json
import os
import pickle
import collections
import contextlib
import warnings
import re
import io
import requests
import pytest
import urllib3
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth, _basic_auth_str
from requests.compat import (
from requests.cookies import (
from requests.exceptions import (
from requests.exceptions import SSLError as RequestsSSLError
from requests.models import PreparedRequest
from requests.structures import CaseInsensitiveDict
from requests.sessions import SessionRedirectMixin
from requests.models import urlencode
from requests.hooks import default_hooks
from requests.compat import JSONDecodeError, is_py3, MutableMapping
from .compat import StringIO, u
from .utils import override_environ
from urllib3.util import Timeout as Urllib3Timeout
@pytest.mark.parametrize('url,expected', (('http://google.com', 'http://google.com/'), (u'http://ジェーピーニック.jp', u'http://xn--hckqz9bzb1cyrb.jp/'), (u'http://xn--n3h.net/', u'http://xn--n3h.net/'), (u'http://ジェーピーニック.jp'.encode('utf-8'), u'http://xn--hckqz9bzb1cyrb.jp/'), (u'http://straße.de/straße', u'http://xn--strae-oqa.de/stra%C3%9Fe'), (u'http://straße.de/straße'.encode('utf-8'), u'http://xn--strae-oqa.de/stra%C3%9Fe'), (u'http://Königsgäßchen.de/straße', u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'), (u'http://Königsgäßchen.de/straße'.encode('utf-8'), u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'), (b'http://xn--n3h.net/', u'http://xn--n3h.net/'), (b'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/', u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/'), (u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/', u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/')))
def test_preparing_url(self, url, expected):

    def normalize_percent_encode(x):
        for c in re.findall('%[a-fA-F0-9]{2}', x):
            x = x.replace(c, c.upper())
        return x
    r = requests.Request('GET', url=url)
    p = r.prepare()
    assert normalize_percent_encode(p.url) == expected