import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
def get_raw_json(self, url, user_agent_headers=None, params=None, proxy=None, timeout=30):
    utils.get_yf_logger().debug(f'get_raw_json(): {url}')
    response = self.get(url, user_agent_headers=user_agent_headers, params=params, proxy=proxy, timeout=timeout)
    response.raise_for_status()
    return response.json()