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
def text_to_be_present_in_element_value(locator: Tuple[str, str], text_: str) -> Callable[[WebDriverOrWebElement], bool]:
    """An expectation for checking if the given text is present in the
    element's value.

    locator, text
    """

    def _predicate(driver: WebDriverOrWebElement):
        try:
            element_text = driver.find_element(*locator).get_attribute('value')
            return text_ in element_text
        except StaleElementReferenceException:
            return False
    return _predicate