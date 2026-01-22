import errno
import logging
import os
import subprocess
import typing
from abc import ABC
from abc import abstractmethod
from io import IOBase
from platform import system
from subprocess import PIPE
from time import sleep
from urllib import request
from urllib.error import URLError
from selenium.common.exceptions import WebDriverException
from selenium.types import SubprocessStdAlias
from selenium.webdriver.common import utils
@property
def service_url(self) -> str:
    """Gets the url of the Service."""
    return f'http://{utils.join_host_port('localhost', self.port)}'