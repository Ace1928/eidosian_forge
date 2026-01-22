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
def element_selection_state_to_be(element: WebElement, is_selected: bool) -> Callable[[Any], bool]:
    """An expectation for checking if the given element is selected.

    element is WebElement object is_selected is a Boolean.
    """

    def _predicate(_):
        return element.is_selected() == is_selected
    return _predicate