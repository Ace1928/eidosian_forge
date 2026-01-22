from concurrent.futures.thread import ThreadPoolExecutor
from traceback import print_tb
import pytest
import portend
import requests
from requests_toolbelt.sessions import BaseUrlSession as Session
from jaraco.context import ExceptionTrap
from cheroot import wsgi
from cheroot._compat import IS_MACOS, IS_WINDOWS
def do_request():
    with ExceptionTrap(requests.exceptions.ConnectionError) as trap:
        resp = session.get('info')
        resp.raise_for_status()
    print_tb(trap.tb)
    return bool(trap)