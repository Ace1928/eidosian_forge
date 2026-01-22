import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
@classmethod
def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
    attrs = [attr.name for attr in dataclasses.fields(cls)]
    engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
    return engine_args