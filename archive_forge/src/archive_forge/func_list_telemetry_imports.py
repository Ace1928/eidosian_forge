import re
import sys
from types import TracebackType
from typing import TYPE_CHECKING, ContextManager, Dict, List, Optional, Set, Type
import wandb
from wandb.proto.wandb_telemetry_pb2 import Imports as TelemetryImports
from wandb.proto.wandb_telemetry_pb2 import TelemetryRecord
def list_telemetry_imports(only_imported: bool=False) -> Set[str]:
    import_telemetry_set = {desc.name for desc in TelemetryImports.DESCRIPTOR.fields if desc.type == desc.TYPE_BOOL}
    if only_imported:
        imported_modules_set = set(sys.modules)
        return imported_modules_set.intersection(import_telemetry_set)
    return import_telemetry_set