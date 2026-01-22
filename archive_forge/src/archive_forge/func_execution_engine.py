from typing import Any, Dict
from uuid import uuid4
from adagio.instances import (
from adagio.specs import WorkflowSpec
from fugue.constants import FUGUE_CONF_WORKFLOW_CONCURRENCY
from fugue.dataframe import DataFrame
from fugue.execution.execution_engine import ExecutionEngine
from fugue.rpc.base import make_rpc_server, RPCServer
from fugue.workflow._checkpoint import CheckpointPath
from triad import SerializableRLock, ParamDict
@property
def execution_engine(self) -> ExecutionEngine:
    return self._fugue_engine