import abc
import mimetypes
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Generic, Callable, Iterable, Optional, Tuple, TypeVar, Union
from .cloudpath import CloudImplementation, CloudPath, implementation_registry
from .enums import FileCacheMode
from .exceptions import InvalidConfigurationException
@classmethod
def get_default_client(cls) -> 'Client':
    """Get the default client, which the one that is used when instantiating a cloud path
        instance for this cloud without a client specified.
        """
    if cls._default_client is None:
        cls._default_client = cls()
    return cls._default_client