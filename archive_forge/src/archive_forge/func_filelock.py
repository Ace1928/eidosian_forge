from __future__ import annotations
import os
import abc
import atexit
import pathlib
import filelock
import contextlib
from lazyops.types import BaseModel, Field
from lazyops.utils.logs import logger
from lazyops.utils.serialization import Json
from typing import Optional, Dict, Any, Set, List, Union, Generator, TYPE_CHECKING
@property
def filelock(self) -> filelock.SoftFileLock:
    """
        Returns the filelock
        """
    if self._filelock is None:
        try:
            self._filelock = filelock.SoftFileLock(self.filelock_path.as_posix(), timeout=self.timeout, thread_local=False)
            with self._filelock.acquire():
                if not self.filepath.exists():
                    self.filepath.write_text('{}')
        except Exception as e:
            from lazyops.libs.logging import logger
            logger.trace(f'Error creating filelock for {self.filepath}', e)
            raise e
    return self._filelock