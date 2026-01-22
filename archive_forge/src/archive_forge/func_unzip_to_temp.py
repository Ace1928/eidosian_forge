import os
import re
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, List, Tuple
import fsspec
import fsspec.core as fc
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
@contextmanager
def unzip_to_temp(fobj: Any) -> Iterator[str]:
    """Unzip a file object into a temporary directory.

    :param fobj: the file object

    .. admonition:: Examples

        .. code-block:: python

            from fugue_ml.utils.io import zip_temp
            from io import BytesIO

            bio = BytesIO()
            with zip_temp(bio) as tmpdir:
                # create files in the tmpdir (string)

            with unzip_to_temp(BytesIO(bio.getvalue())) as tmpdir:
                # read files from the tmpdir (string)
    """
    if isinstance(fobj, str):
        with fsspec.open(fobj, 'rb') as f:
            with unzip_to_temp(f) as tmpdir:
                yield tmpdir
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            with zipfile.ZipFile(fobj, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)
            yield tmpdirname