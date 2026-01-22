import json
import logging
from typing import Dict, NamedTuple, Optional, Union
import urllib
from absl import flags
from utils import bq_consts
from utils import bq_error
def get_discovery_url_from_root_url(root_url: str, api_version: str='v2') -> str:
    """Returns the discovery doc URL from a root URL."""
    parts = urllib.parse.urlsplit(root_url)
    query = urllib.parse.urlencode({'version': api_version})
    parts = parts._replace(path='/$discovery/rest', query=query)
    return urllib.parse.urlunsplit(parts)