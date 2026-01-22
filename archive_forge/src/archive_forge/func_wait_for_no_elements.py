import os
import sys
import time
import logging
import warnings
import percy
import requests
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
from dash.testing.wait import (
from dash.testing.dash_page import DashPageMixin
from dash.testing.errors import DashAppLoadingError, BrowserError, TestingTimeoutError
from dash.testing.consts import SELENIUM_GRID_DEFAULT
def wait_for_no_elements(self, selector, timeout=None):
    """Explicit wait until an element is NOT found. timeout defaults to
        the fixture's `wait_timeout`."""
    until(lambda: self.driver.execute_script(f"return document.querySelectorAll('{selector}').length") == 0, timeout or self._wait_timeout)