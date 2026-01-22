import re
from collections.abc import Iterable
from typing import Any
from typing import Callable
from typing import List
from typing import Literal
from typing import Tuple
from typing import TypeVar
from typing import Union
from selenium.common.exceptions import NoAlertPresentException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoSuchFrameException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webdriver import WebElement
def element_located_to_be_selected(locator: Tuple[str, str]) -> Callable[[WebDriverOrWebElement], bool]:
    """An expectation for the element to be located is selected.

    locator is a tuple of (by, path)
    """

    def _predicate(driver: WebDriverOrWebElement):
        return driver.find_element(*locator).is_selected()
    return _predicate