import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime
from queue import Queue
from typing import (
import requests
import wandb
from wandb import util
from wandb.errors import CommError, UsageError
from wandb.errors.util import ProtobufErrorHandler
from wandb.filesync.dir_watcher import DirWatcher
from wandb.proto import wandb_internal_pb2
from wandb.sdk.artifacts.artifact_saver import ArtifactSaver
from wandb.sdk.interface import interface
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import (
from wandb.sdk.internal.file_pusher import FilePusher
from wandb.sdk.internal.job_builder import JobBuilder
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import (
from wandb.sdk.lib.mailbox import ContextCancelledError
from wandb.sdk.lib.proto_util import message_to_dict
def send_request_log_artifact(self, record: 'Record') -> None:
    assert record.control.req_resp
    result = proto_util._result_from_record(record)
    artifact = record.request.log_artifact.artifact
    history_step = record.request.log_artifact.history_step
    try:
        res = self._send_artifact(artifact, history_step)
        assert res, 'Unable to send artifact'
        result.response.log_artifact_response.artifact_id = res['id']
        logger.info(f'logged artifact {artifact.name} - {res}')
    except Exception as e:
        result.response.log_artifact_response.error_message = f'error logging artifact "{artifact.type}/{artifact.name}": {e}'
    self._respond_result(result)