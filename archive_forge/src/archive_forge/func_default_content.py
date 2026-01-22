from typing import Optional
from typing import Union
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoSuchFrameException
from selenium.common.exceptions import NoSuchWindowException
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from .command import Command
def default_content(self) -> None:
    """Switch focus to the default frame.

        :Usage:
            ::

                driver.switch_to.default_content()
        """
    self._driver.execute(Command.SWITCH_TO_FRAME, {'id': None})