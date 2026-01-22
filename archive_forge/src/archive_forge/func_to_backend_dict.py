import json
from typing import Any, Dict, NewType, Optional, Sequence
from wandb.proto import wandb_internal_pb2
from wandb.sdk.lib import proto_util, telemetry
def to_backend_dict(self, telemetry_record: telemetry.TelemetryRecord, framework: Optional[str], start_time_millis: int, metric_pbdicts: Sequence[Dict[int, Any]]) -> BackendConfigDict:
    """Returns a dictionary representation expected by the backend.

        The backend expects the configuration in a specific format, and the
        config is also used to store additional metadata about the run.

        Args:
            telemetry_record: Telemetry information to insert.
            framework: The detected framework used in the run (e.g. TensorFlow).
            start_time_millis: The run's start time in Unix milliseconds.
            metric_pbdicts: List of dict representations of metric protobuffers.
        """
    backend_dict = self._tree.copy()
    wandb_internal = backend_dict.setdefault(_WANDB_INTERNAL_KEY, {})
    py_version = telemetry_record.python_version
    if py_version:
        wandb_internal['python_version'] = py_version
    cli_version = telemetry_record.cli_version
    if cli_version:
        wandb_internal['cli_version'] = cli_version
    if framework:
        wandb_internal['framework'] = framework
    huggingface_version = telemetry_record.huggingface_version
    if huggingface_version:
        wandb_internal['huggingface_version'] = huggingface_version
    wandb_internal['is_jupyter_run'] = telemetry_record.env.jupyter
    wandb_internal['is_kaggle_kernel'] = telemetry_record.env.kaggle
    wandb_internal['start_time'] = start_time_millis
    wandb_internal['t'] = proto_util.proto_encode_to_dict(telemetry_record)
    if metric_pbdicts:
        wandb_internal['m'] = metric_pbdicts
    return BackendConfigDict({key: {'desc': None, 'value': value} for key, value in self._tree.items()})