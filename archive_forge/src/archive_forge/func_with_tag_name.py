from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def with_tag_name(tag_name: str) -> 'RelativeBy':
    """Start searching for relative objects using a tag name.

    Note: This method may be removed in future versions, please use
    `locate_with` instead.
    :Args:
        - tag_name: the DOM tag of element to start searching.
    :Returns:
        - RelativeBy - use this object to create filters within a
            `find_elements` call.
    """
    if not tag_name:
        raise WebDriverException('tag_name can not be null')
    return RelativeBy({'css selector': tag_name})