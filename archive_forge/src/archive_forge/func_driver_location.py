import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List
from selenium.common import WebDriverException
from selenium.webdriver.common.options import BaseOptions
def driver_location(self, options: BaseOptions) -> str:
    """Determines the path of the correct driver.

        :Args:
         - browser: which browser to get the driver path for.
        :Returns: The driver path to use
        """
    browser = options.capabilities['browserName']
    args = [str(self.get_binary()), '--browser', browser]
    if options.browser_version:
        args.append('--browser-version')
        args.append(str(options.browser_version))
    binary_location = getattr(options, 'binary_location', None)
    if binary_location:
        args.append('--browser-path')
        args.append(str(binary_location))
    proxy = options.proxy
    if proxy and (proxy.http_proxy or proxy.ssl_proxy):
        args.append('--proxy')
        value = proxy.ssl_proxy if proxy.ssl_proxy else proxy.http_proxy
        args.append(value)
    output = self.run(args)
    browser_path = output['browser_path']
    driver_path = output['driver_path']
    logger.debug('Using driver at: %s', driver_path)
    if hasattr(options.__class__, 'binary_location') and browser_path:
        options.binary_location = browser_path
        options.browser_version = None
    return driver_path