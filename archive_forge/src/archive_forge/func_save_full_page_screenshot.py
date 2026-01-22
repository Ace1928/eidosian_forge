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
def save_full_page_screenshot(self, filename) -> bool:
    """Saves a full document screenshot of the current window to a PNG
        image file. Returns False if there is any IOError, else returns True.
        Use full paths in your filename.

        :Args:
         - filename: The full path you wish to save your screenshot to. This
           should end with a `.png` extension.

        :Usage:
            ::

                driver.save_full_page_screenshot('/Screenshots/foo.png')
        """
    return self.get_full_page_screenshot_as_file(filename)