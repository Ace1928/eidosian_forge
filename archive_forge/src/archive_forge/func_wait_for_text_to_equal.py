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
def wait_for_text_to_equal(self, selector, text, timeout=None):
    """Explicit wait until the element's text equals the expected `text`.

        timeout if not set, equals to the fixture's `wait_timeout`
        shortcut to `WebDriverWait` with customized `text_to_equal`
        condition.
        """
    method = text_to_equal(selector, text, timeout or self.wait_timeout)
    return self._wait_for(method=method, timeout=timeout, msg=method.message)