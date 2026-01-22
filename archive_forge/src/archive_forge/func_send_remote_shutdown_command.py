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
def send_remote_shutdown_command(self) -> None:
    """Dispatch an HTTP request to the shutdown endpoint for the service in
        an attempt to stop it."""
    try:
        request.urlopen(f'{self.service_url}/shutdown')
    except URLError:
        return
    for _ in range(30):
        if not self.is_connectable():
            break
        sleep(1)