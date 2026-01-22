import os
import re
import urllib
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri, load_class
from ray._private.auto_init_hook import wrap_auto_init
def delete_dir(self, path: str) -> bool:
    """Delete a directory and its contents, recursively.

        Examples:
            .. testcode::

                import ray
                from ray._private import storage

                ray.shutdown()

                ray.init(storage="/tmp/storage/cluster_1/storage")

                client = storage.get_client("my_app")
                client.put("path/foo.txt", b"bar")
                assert client.delete_dir("path")

        Args:
            path: Relative directory of the blob.

        Returns:
            Whether the dir was deleted.
        """
    full_path = self._resolve_path(path)
    try:
        self.fs.delete_dir(full_path)
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        if _is_os_error_file_not_found(e):
            return False
        raise e