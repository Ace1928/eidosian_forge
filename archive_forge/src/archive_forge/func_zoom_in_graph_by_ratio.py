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
def zoom_in_graph_by_ratio(self, elem_or_selector, start_fraction=0.5, zoom_box_fraction=0.2, compare=True):
    """Zoom out a graph with a zoom box fraction of component dimension
        default start at middle with a rectangle of 1/5 of the dimension use
        `compare` to control if we check the svg get changed."""
    elem = self._get_element(elem_or_selector)
    prev = elem.get_attribute('innerHTML')
    w, h = (elem.size['width'], elem.size['height'])
    try:
        ActionChains(self.driver).move_to_element_with_offset(elem, w * start_fraction, h * start_fraction).drag_and_drop_by_offset(elem, w * zoom_box_fraction, h * zoom_box_fraction).perform()
    except MoveTargetOutOfBoundsException:
        logger.exception('graph offset outside of the boundary')
    if compare:
        assert prev != elem.get_attribute('innerHTML'), 'SVG content should be different after zoom'