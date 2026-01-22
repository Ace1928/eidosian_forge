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
def requestJson(self, verb: str, url: str, parameters: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str, Any]]=None, input: Optional[Any]=None, cnx: Optional[Union[HTTPRequestsConnectionClass, HTTPSRequestsConnectionClass]]=None) -> Tuple[int, Dict[str, Any], str]:

    def encode(input: Any) -> Tuple[str, str]:
        return ('application/json', json.dumps(input))
    return self.__requestEncode(cnx, verb, url, parameters, headers, input, encode)