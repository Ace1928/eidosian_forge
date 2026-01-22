from __future__ import annotations
import logging # isort:skip
from ..util.dependencies import import_required # isort:skip
import_required("selenium.webdriver",
import atexit
import os
from os.path import devnull
from shutil import which
from typing import TYPE_CHECKING, Literal
from packaging.version import Version
from ..settings import settings
def scale_factor_less_than_web_driver_device_pixel_ratio(scale_factor: float, web_driver: WebDriver) -> bool:
    device_pixel_ratio = get_web_driver_device_pixel_ratio(web_driver)
    return device_pixel_ratio >= scale_factor