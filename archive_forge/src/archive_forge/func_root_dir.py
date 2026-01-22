import os
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union
from packaging import version
from typing_extensions import override
import wandb
from wandb import Artifact
from wandb.sdk.lib import RunDisabled, telemetry
from wandb.sdk.wandb_run import Run
@property
def root_dir(self) -> Optional[str]:
    """Return the root directory.

        Return the root directory where all versions of an experiment get saved, or `None` if the logger does not
        save data locally.
        """
    return self.save_dir.parent if self.save_dir else None