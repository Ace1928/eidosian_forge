import abc
import logging
import os
import random
import shutil
import time
import urllib
import uuid
from collections import namedtuple
from typing import IO, List, Optional, Tuple
import ray
from ray._private.ray_constants import DEFAULT_OBJECT_PREFIX
from ray._raylet import ObjectRef
def create_url_with_offset(*, url: str, offset: int, size: int) -> str:
    """Methods to create a URL with offset.

    When ray spills objects, it fuses multiple objects
    into one file to optimize the performance. That says, each object
    needs to keep tracking of its own special url to store metadata.

    This method creates an url_with_offset, which is used internally
    by Ray.

    Created url_with_offset can be passed to the self._get_base_url method
    to parse the filename used to store files.

    Example) file://path/to/file?offset=""&size=""

    Args:
        url: url to the object stored in the external storage.
        offset: Offset from the beginning of the file to
            the first bytes of this object.
        size: Size of the object that is stored in the url.
            It is used to calculate the last offset.

    Returns:
        url_with_offset stored internally to find
        objects from external storage.
    """
    return f'{url}?offset={offset}&size={size}'