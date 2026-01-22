import git
from git.exc import InvalidGitRepositoryError
from git.config import GitConfigParser
from io import BytesIO
import weakref
from typing import Any, Sequence, TYPE_CHECKING, Union
from git.types import PathLike
def flush_to_index(self) -> None:
    """Flush changes in our configuration file to the index."""
    assert self._smref is not None
    assert not isinstance(self._file_or_files, BytesIO)
    sm = self._smref()
    if sm is not None:
        index = self._index
        if index is None:
            index = sm.repo.index
        index.add([sm.k_modules_file], write=self._auto_write)
        sm._clear_cache()