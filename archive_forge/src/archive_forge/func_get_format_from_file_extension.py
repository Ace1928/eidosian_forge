from __future__ import annotations
import warnings
import typing
from typing import Any, Collection, Optional, Protocol, TypeVar
import google.protobuf.json_format
import google.protobuf.message
import google.protobuf.text_format
import onnx
def get_format_from_file_extension(self, file_extension: str) -> str | None:
    """Get the corresponding format from a file extension.

        Args:
            file_extension: The file extension to get a format for.

        Returns:
            The format for the file extension, or None if not found.
        """
    return self._extension_to_format.get(file_extension)