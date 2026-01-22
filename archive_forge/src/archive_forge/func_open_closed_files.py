import argparse
import errno
import glob
import logging
import logging.handlers
import os
import platform
import re
import shutil
import time
import traceback
from typing import Callable, List, Optional, Set
from ray._raylet import GcsClient
import ray._private.ray_constants as ray_constants
import ray._private.services as services
import ray._private.utils
from ray._private.ray_logging import setup_component_logger
def open_closed_files(self):
    """Open some closed files if they may have new lines.

        Opening more files may require us to close some of the already open
        files.
        """
    if not self.can_open_more_files:
        self._close_all_files()
    files_with_no_updates = []
    while len(self.closed_file_infos) > 0:
        if len(self.open_file_infos) >= self.max_files_open:
            self.can_open_more_files = False
            break
        file_info = self.closed_file_infos.pop(0)
        assert file_info.file_handle is None
        try:
            file_size = os.path.getsize(file_info.filename)
        except (IOError, OSError) as e:
            if e.errno == errno.ENOENT:
                logger.warning(f'Warning: The file {file_info.filename} was not found.')
                self.log_filenames.remove(file_info.filename)
                continue
            raise e
        if file_size > file_info.size_when_last_opened:
            try:
                f = open(file_info.filename, 'rb')
            except (IOError, OSError) as e:
                if e.errno == errno.ENOENT:
                    logger.warning(f'Warning: The file {file_info.filename} was not found.')
                    self.log_filenames.remove(file_info.filename)
                    continue
                else:
                    raise e
            f.seek(file_info.file_position)
            file_info.size_when_last_opened = file_size
            file_info.file_handle = f
            self.open_file_infos.append(file_info)
        else:
            files_with_no_updates.append(file_info)
    if len(self.open_file_infos) >= self.max_files_open:
        self.can_open_more_files = False
    self.closed_file_infos += files_with_no_updates