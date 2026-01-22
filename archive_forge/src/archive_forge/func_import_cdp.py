import json
import pkgutil
from contextlib import asynccontextmanager
from importlib import import_module
from selenium.webdriver.common.by import By
def import_cdp():
    global cdp
    if not cdp:
        cdp = import_module('selenium.webdriver.common.bidi.cdp')