import email.message
import logging
import pathlib
import traceback
import urllib.parse
import warnings
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Tuple, Type, Union
import requests
from gitlab import types
def get_content_type(content_type: Optional[str]) -> str:
    message = email.message.Message()
    message['content-type'] = content_type
    return message.get_content_type()