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
def url_contains(url: str) -> Callable[[WebDriver], bool]:
    """An expectation for checking that the current url contains a case-
    sensitive substring.

    url is the fragment of url expected, returns True when the url
    matches, False otherwise
    """

    def _predicate(driver: WebDriver):
        return url in driver.current_url
    return _predicate