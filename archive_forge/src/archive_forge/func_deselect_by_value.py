from typing import List
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def deselect_by_value(self, value: str) -> None:
    """Deselect all options that have a value matching the argument. That
        is, when given "foo" this would deselect an option like:

         <option value="foo">Bar</option>

        :Args:
         - value - The value to match against

         throws NoSuchElementException If there is no option with specified value in SELECT
        """
    if not self.is_multiple:
        raise NotImplementedError('You may only deselect options of a multi-select')
    matched = False
    css = f'option[value = {self._escape_string(value)}]'
    opts = self._el.find_elements(By.CSS_SELECTOR, css)
    for opt in opts:
        self._unset_selected(opt)
        matched = True
    if not matched:
        raise NoSuchElementException(f'Could not locate element with value: {value}')