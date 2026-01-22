import typing
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from .abstract_event_listener import AbstractEventListener
@property
def wrapped_element(self) -> WebElement:
    """Returns the WebElement wrapped by this EventFiringWebElement
        instance."""
    return self._webelement