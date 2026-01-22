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
@classmethod
def resetConnectionClasses(cls) -> None:
    cls.__persist = True
    cls.__httpConnectionClass = HTTPRequestsConnectionClass
    cls.__httpsConnectionClass = HTTPSRequestsConnectionClass