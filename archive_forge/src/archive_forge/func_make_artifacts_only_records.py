import json
import logging
import math
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import numpy as np
from google.protobuf.json_format import ParseDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from wandb import Artifact
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_settings_pb2
from wandb.proto import wandb_telemetry_pb2 as telem_pb
from wandb.sdk.interface.interface import file_policy_to_enum
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import context
from wandb.sdk.internal.sender import SendManager
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.util import coalesce, recursive_cast_dictlike_to_dict
from .protocols import ImporterRun
def make_artifacts_only_records(self, artifacts: Optional[Iterable[Artifact]]=None, used_artifacts: Optional[Iterable[Artifact]]=None) -> Iterable[pb.Record]:
    """Only make records required to upload artifacts.

        Escape hatch for adding extra artifacts to a run.
        """
    yield self._make_run_record()
    if used_artifacts:
        for art in used_artifacts:
            yield self._make_artifact_record(art, use_artifact=True)
    if artifacts:
        for art in artifacts:
            yield self._make_artifact_record(art)