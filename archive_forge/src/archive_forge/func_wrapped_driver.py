import typing
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from .abstract_event_listener import AbstractEventListener
@property
def wrapped_driver(self) -> WebDriver:
    """Returns the WebDriver instance wrapped by this
        EventsFiringWebDriver."""
    return self._driver