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
def set_window_rect(self, x=None, y=None, width=None, height=None) -> dict:
    """Sets the x, y coordinates of the window as well as height and width
        of the current window. This method is only supported for W3C compatible
        browsers; other browsers should use `set_window_position` and
        `set_window_size`.

        :Usage:
            ::

                driver.set_window_rect(x=10, y=10)
                driver.set_window_rect(width=100, height=200)
                driver.set_window_rect(x=10, y=10, width=100, height=200)
        """
    if (x is None and y is None) and (not height and (not width)):
        raise InvalidArgumentException('x and y or height and width need values')
    return self.execute(Command.SET_WINDOW_RECT, {'x': x, 'y': y, 'width': width, 'height': height})['value']