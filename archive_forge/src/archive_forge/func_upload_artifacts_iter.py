import logging
import math
import os
import posixpath
from abc import abstractmethod
from collections import namedtuple
from concurrent.futures import as_completed
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils import chunk_list
from mlflow.utils.file_utils import (
from mlflow.utils.uri import is_fuse_or_uc_volumes_uri
def upload_artifacts_iter():
    for staged_upload_chunk in chunk_list(staged_uploads, _ARTIFACT_UPLOAD_BATCH_SIZE):
        write_credential_infos = self._get_write_credential_infos(remote_file_paths=[staged_upload.artifact_file_path for staged_upload in staged_upload_chunk])
        inflight_uploads = {}
        for staged_upload, write_credential_info in zip(staged_upload_chunk, write_credential_infos):
            upload_future = self.thread_pool.submit(self._upload_to_cloud, cloud_credential_info=write_credential_info, src_file_path=staged_upload.src_file_path, artifact_file_path=staged_upload.artifact_file_path)
            inflight_uploads[staged_upload.src_file_path] = upload_future
        yield from inflight_uploads.items()