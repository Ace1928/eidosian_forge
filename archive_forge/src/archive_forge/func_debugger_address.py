import base64
import os
from typing import BinaryIO
from typing import List
from typing import Union
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions
@debugger_address.setter
def debugger_address(self, value: str) -> None:
    """Allows you to set the address of the remote devtools instance that
        the ChromeDriver instance will try to connect to during an active wait.

        :Args:
         - value: address of remote devtools instance if any (hostname[:port])
        """
    if not isinstance(value, str):
        raise TypeError('Debugger Address must be a string')
    self._debugger_address = value