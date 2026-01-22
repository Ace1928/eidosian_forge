import collections
from enum import Enum
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
def tag_storage_type(storage: 'StorageContext'):
    """Records the storage configuration of an experiment.

    The storage configuration is set by `RunConfig(storage_path, storage_filesystem)`.

    The possible storage types (defined by `pyarrow.fs.FileSystem.type_name`) are:
    - 'local' = pyarrow.fs.LocalFileSystem. This includes NFS usage.
    - 'mock' = pyarrow.fs._MockFileSystem. This is used for testing.
    - ('s3', 'gcs', 'abfs', 'hdfs'): Various remote storage schemes
        with default implementations in pyarrow.
    - 'custom' = All other storage schemes, which includes ALL cases where a
        custom `storage_filesystem` is provided.
    - 'other' = catches any other cases not explicitly handled above.
    """
    whitelist = {'local', 'mock', 's3', 'gcs', 'abfs', 'hdfs'}
    if storage.custom_fs_provided:
        storage_config_tag = 'custom'
    elif storage.storage_filesystem.type_name in whitelist:
        storage_config_tag = storage.storage_filesystem.type_name
    else:
        storage_config_tag = 'other'
    record_extra_usage_tag(TagKey.AIR_STORAGE_CONFIGURATION, storage_config_tag)