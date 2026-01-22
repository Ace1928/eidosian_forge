from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def to_right_of(self, element_or_locator: Union[WebElement, Dict]=None) -> 'RelativeBy':
    """Add a filter to look for elements right of.

        :Args:
            - element_or_locator: Element to look right of
        """
    if not element_or_locator:
        raise WebDriverException('Element or locator must be given when calling to_right_of method')
    self.filters.append({'kind': 'right', 'args': [element_or_locator]})
    return self