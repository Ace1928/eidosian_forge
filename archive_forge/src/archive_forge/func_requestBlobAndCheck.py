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
def requestBlobAndCheck(self, verb: str, url: str, parameters: Optional[Dict[str, str]]=None, headers: Optional[Dict[str, str]]=None, input: Optional[str]=None, cnx: Optional[Union[HTTPRequestsConnectionClass, HTTPSRequestsConnectionClass]]=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return self.__check(*self.requestBlob(verb, url, parameters, headers, input, self.__customConnection(url)))