import io
import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from io import IOBase
from typing import (
import requests
import requests.adapters
from urllib3 import Retry
import github.Consts as Consts
import github.GithubException as GithubException
def graphql_query(self, query: str, variables: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
        :calls: `POST /graphql <https://docs.github.com/en/graphql>`_
        """
    input_ = {'query': query, 'variables': {'input': variables}}
    response_headers, data = self.requestJsonAndCheck('POST', self.graphql_url, input=input_)
    if 'errors' in data:
        raise self.createException(400, response_headers, data)
    return (response_headers, data)