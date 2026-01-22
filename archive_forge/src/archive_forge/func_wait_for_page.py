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
def wait_for_page(self, url=None, timeout=10):
    """wait_for_page navigates to the url in webdriver wait until the
        renderer is loaded in browser.

        use the `server_url` if url is not provided.
        """
    self.driver.get(self.server_url if url is None else url)
    try:
        self.wait_for_element_by_css_selector(self.dash_entry_locator, timeout=timeout)
    except TimeoutException as exc:
        logger.exception('dash server is not loaded within %s seconds', timeout)
        logs = '\n'.join((str(log) for log in self.get_logs()))
        logger.debug(logs)
        html = self.find_element('body').get_property('innerHTML')
        raise DashAppLoadingError(f'the expected Dash react entry point cannot be loaded in browser\n HTML => {html}\n Console Logs => {logs}\n') from exc
    if self._pause:
        import pdb
        pdb.set_trace()