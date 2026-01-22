import requests
import warnings
from requests import adapters
from requests import sessions
from .. import exceptions as exc
from .._compat import gaecontrib
from .._compat import timeout
def monkeypatch(validate_certificate=True):
    """Sets up all Sessions to use AppEngineAdapter by default.

    If you don't want to deal with configuring your own Sessions,
    or if you use libraries that use requests directly (ie requests.post),
    then you may prefer to monkeypatch and auto-configure all Sessions.

    .. warning: :

        If ``validate_certificate`` is ``False``, certification validation will
        effectively be disabled for all requests.
    """
    _check_version()
    adapter = AppEngineAdapter
    if not validate_certificate:
        adapter = InsecureAppEngineAdapter
    sessions.HTTPAdapter = adapter
    adapters.HTTPAdapter = adapter