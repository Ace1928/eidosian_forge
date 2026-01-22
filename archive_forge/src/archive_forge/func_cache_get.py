import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
@lru_cache_freezeargs
@lru_cache(maxsize=cache_maxsize)
def cache_get(self, url, user_agent_headers=None, params=None, proxy=None, timeout=30):
    return self.get(url, user_agent_headers, params, proxy, timeout)