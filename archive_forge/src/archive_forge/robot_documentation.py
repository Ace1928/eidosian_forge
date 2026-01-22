from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import fetch_url, open_url
import json
import time

    Make general request to Hetzner's JSON robot API, with retries until a condition is satisfied.

    The condition is tested by calling ``check_done_callback(result, error)``. If it is not satisfied,
    it will be retried with delays ``check_done_delay`` (in seconds) until a total timeout of
    ``check_done_timeout`` (in seconds) since the time the first request is started is reached.

    If ``skip_first`` is specified, will assume that a first call has already been made and will
    directly start with waiting.
    