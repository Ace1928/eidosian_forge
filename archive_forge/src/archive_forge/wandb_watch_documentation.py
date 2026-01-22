import logging
from typing import Optional
import wandb
from .lib import telemetry
Remove pytorch model topology, gradient and parameter hooks.

    Args:
        models: (list) Optional list of pytorch models that have had watch called on them
    