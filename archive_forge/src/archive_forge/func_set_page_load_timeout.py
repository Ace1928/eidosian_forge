import base64
import contextlib
import copy
import os
import pkgutil
import types
import typing
import warnings
import zipfile
from abc import ABCMeta
from base64 import b64decode
from base64 import urlsafe_b64encode
from contextlib import asynccontextmanager
from contextlib import contextmanager
from importlib import import_module
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from selenium.common.exceptions import InvalidArgumentException
from selenium.common.exceptions import JavascriptException
from selenium.common.exceptions import NoSuchCookieException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.options import BaseOptions
from selenium.webdriver.common.print_page_options import PrintOptions
from selenium.webdriver.common.timeouts import Timeouts
from selenium.webdriver.common.virtual_authenticator import Credential
from selenium.webdriver.common.virtual_authenticator import VirtualAuthenticatorOptions
from selenium.webdriver.common.virtual_authenticator import (
from selenium.webdriver.support.relative_locator import RelativeBy
from .bidi_connection import BidiConnection
from .command import Command
from .errorhandler import ErrorHandler
from .file_detector import FileDetector
from .file_detector import LocalFileDetector
from .mobile import Mobile
from .remote_connection import RemoteConnection
from .script_key import ScriptKey
from .shadowroot import ShadowRoot
from .switch_to import SwitchTo
from .webelement import WebElement
def set_page_load_timeout(self, time_to_wait: float) -> None:
    """Set the amount of time to wait for a page load to complete before
        throwing an error.

        :Args:
         - time_to_wait: The amount of time to wait

        :Usage:
            ::

                driver.set_page_load_timeout(30)
        """
    try:
        self.execute(Command.SET_TIMEOUTS, {'pageLoad': int(float(time_to_wait) * 1000)})
    except WebDriverException:
        self.execute(Command.SET_TIMEOUTS, {'ms': float(time_to_wait) * 1000, 'type': 'page load'})