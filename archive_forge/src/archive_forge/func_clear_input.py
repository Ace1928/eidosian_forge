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
def clear_input(self, elem_or_selector):
    """Simulate key press to clear the input."""
    elem = self._get_element(elem_or_selector)
    logger.debug('clear input with %s => %s', elem_or_selector, elem)
    ActionChains(self.driver).move_to_element(elem).pause(0.2).click(elem).send_keys(Keys.END).key_down(Keys.SHIFT).send_keys(Keys.HOME).key_up(Keys.SHIFT).send_keys(Keys.DELETE).perform()