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
def select_dcc_dropdown(self, elem_or_selector, value=None, index=None):
    dropdown = self._get_element(elem_or_selector)
    dropdown.click()
    menu = dropdown.find_element(By.CSS_SELECTOR, 'div.Select-menu-outer')
    logger.debug('the available options are %s', '|'.join(menu.text.split('\n')))
    options = menu.find_elements(By.CSS_SELECTOR, 'div.VirtualizedSelectOption')
    if options:
        if isinstance(index, int):
            options[index].click()
            return
        for option in options:
            if option.text == value:
                option.click()
                return
    logger.error('cannot find matching option using value=%s or index=%s', value, index)