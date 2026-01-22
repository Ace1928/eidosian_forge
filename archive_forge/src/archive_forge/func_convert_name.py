import time
from sys import platform
from typing import (
def convert_name(node_name: Optional[str], has_click_handler: Optional[bool]) -> str:
    if node_name == 'a':
        return 'link'
    if node_name == 'input':
        return 'input'
    if node_name == 'img':
        return 'img'
    if node_name == 'button' or has_click_handler:
        return 'button'
    else:
        return 'text'