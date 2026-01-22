from typing import List
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def deselect_all(self) -> None:
    """Clear all selected entries.

        This is only valid when the SELECT supports multiple selections.
        throws NotImplementedError If the SELECT does not support
        multiple selections
        """
    if not self.is_multiple:
        raise NotImplementedError('You may only deselect all options of a multi-select')
    for opt in self.options:
        self._unset_selected(opt)