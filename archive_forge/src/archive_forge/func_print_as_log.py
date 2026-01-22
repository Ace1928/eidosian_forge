import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
def print_as_log(*args: Any) -> None:
    httpclient_log.log(logging.DEBUG, ' '.join(args))