import contextlib
import functools
import os
import sys
from typing import TYPE_CHECKING, List, Optional, Type, cast
from pip._internal.utils.misc import strtobool
from .base import BaseDistribution, BaseEnvironment, FilesystemWheel, MemoryWheel, Wheel
def get_metadata_distribution(metadata_contents: bytes, filename: str, canonical_name: str) -> BaseDistribution:
    """Get the dist representation of the specified METADATA file contents.

    This returns a Distribution instance from the chosen backend sourced from the data
    in `metadata_contents`.

    :param metadata_contents: Contents of a METADATA file within a dist, or one served
                              via PEP 658.
    :param filename: Filename for the dist this metadata represents.
    :param canonical_name: Normalized project name of the given dist.
    """
    return select_backend().Distribution.from_metadata_file_contents(metadata_contents, filename, canonical_name)