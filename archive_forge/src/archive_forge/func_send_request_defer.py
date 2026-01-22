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
def send_request_defer(self, record: 'Record') -> None:
    defer = record.request.defer
    state = defer.state
    logger.info(f'handle sender defer: {state}')

    def transition_state() -> None:
        state = defer.state + 1
        logger.info(f'send defer: {state}')
        self._interface.publish_defer(state)
    done = False
    if state == defer.BEGIN:
        transition_state()
    elif state == defer.FLUSH_RUN:
        self._flush_run()
        transition_state()
    elif state == defer.FLUSH_STATS:
        transition_state()
    elif state == defer.FLUSH_PARTIAL_HISTORY:
        transition_state()
    elif state == defer.FLUSH_TB:
        transition_state()
    elif state == defer.FLUSH_SUM:
        transition_state()
    elif state == defer.FLUSH_DEBOUNCER:
        self.debounce(final=True)
        transition_state()
    elif state == defer.FLUSH_OUTPUT:
        self._output_raw_finish()
        transition_state()
    elif state == defer.FLUSH_JOB:
        self._flush_job()
        transition_state()
    elif state == defer.FLUSH_DIR:
        if self._dir_watcher:
            self._dir_watcher.finish()
            self._dir_watcher = None
        transition_state()
    elif state == defer.FLUSH_FP:
        if self._pusher:
            self._pusher.finish(transition_state)
        else:
            transition_state()
    elif state == defer.JOIN_FP:
        if self._pusher:
            self._pusher.join()
        transition_state()
    elif state == defer.FLUSH_FS:
        if self._fs:
            self._fs.finish(self._exit_code)
            self._fs = None
        transition_state()
    elif state == defer.FLUSH_FINAL:
        self._interface.publish_final()
        self._interface.publish_footer()
        transition_state()
    elif state == defer.END:
        done = True
    else:
        raise AssertionError('unknown state')
    if not done:
        return
    exit_result = wandb_internal_pb2.RunExitResult()
    self._exit_result = exit_result
    if self._record_exit and self._record_exit.control.mailbox_slot:
        result = proto_util._result_from_record(self._record_exit)
        result.exit_result.CopyFrom(exit_result)
        self._respond_result(result)