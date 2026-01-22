import functools
import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import __import__
from typing import Dict, List, Optional, Set, Union
from weakref import WeakSet
import torch._guards
import torch.distributed as dist
def getArtifactLogger(module_qname, artifact_name):
    if artifact_name not in log_registry.artifact_names:
        raise ValueError(f'Artifact name: {repr(artifact_name)} not registered,please call register_artifact({repr(artifact_name)}) in torch._logging.registrations.')
    qname = module_qname + f'.__{artifact_name}'
    log = logging.getLogger(qname)
    log.artifact_name = artifact_name
    log_registry.register_artifact_log(qname)
    configure_artifact_log(log)
    return log