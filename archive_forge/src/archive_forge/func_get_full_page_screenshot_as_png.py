import base64
import logging
import os
import warnings
import zipfile
from contextlib import contextmanager
from io import BytesIO
from selenium.webdriver.common.driver_finder import DriverFinder
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver
from .options import Options
from .remote_connection import FirefoxRemoteConnection
from .service import Service
def get_full_page_screenshot_as_png(self) -> bytes:
    """Gets the full document screenshot of the current window as a binary
        data.

        :Usage:
            ::

                driver.get_full_page_screenshot_as_png()
        """
    return base64.b64decode(self.get_full_page_screenshot_as_base64().encode('ascii'))