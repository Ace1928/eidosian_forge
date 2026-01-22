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
def invisibility_of_element(element: Union[WebElement, Tuple[str, str]]) -> Callable[[WebDriverOrWebElement], Union[WebElement, bool]]:
    """An Expectation for checking that an element is either invisible or not
    present on the DOM.

    element is either a locator (text) or an WebElement
    """
    return invisibility_of_element_located(element)