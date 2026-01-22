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
def send_link_artifact(self, record: 'Record') -> None:
    link = record.link_artifact
    client_id = link.client_id
    server_id = link.server_id
    portfolio_name = link.portfolio_name
    entity = link.portfolio_entity
    project = link.portfolio_project
    aliases = link.portfolio_aliases
    logger.debug(f'link_artifact params - client_id={client_id}, server_id={server_id}, pfolio={portfolio_name}, entity={entity}, project={project}')
    if (client_id or server_id) and portfolio_name and entity and project:
        try:
            self._api.link_artifact(client_id, server_id, portfolio_name, entity, project, aliases)
        except Exception as e:
            logger.warning('Failed to link artifact to portfolio: %s', e)