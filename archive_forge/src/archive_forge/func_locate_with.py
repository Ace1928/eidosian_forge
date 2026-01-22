from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def locate_with(by: By, using: str) -> 'RelativeBy':
    """Start searching for relative objects your search criteria with By.

    :Args:
        - by: The value from `By` passed in.
        - using: search term to find the element with.
    :Returns:
        - RelativeBy - use this object to create filters within a
            `find_elements` call.
    """
    assert by is not None, 'Please pass in a by argument'
    assert using is not None, 'Please pass in a using argument'
    return RelativeBy({by: using})