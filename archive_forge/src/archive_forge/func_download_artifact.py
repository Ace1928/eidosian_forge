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
@staticmethod
@rank_zero_only
def download_artifact(artifact: str, save_dir: Optional[_PATH]=None, artifact_type: Optional[str]=None, use_artifact: Optional[bool]=True) -> str:
    """Downloads an artifact from the wandb server.

        Args:
            artifact: The path of the artifact to download.
            save_dir: The directory to save the artifact to.
            artifact_type: The type of artifact to download.
            use_artifact: Whether to add an edge between the artifact graph.

        Returns:
            The path to the downloaded artifact.

        """
    if wandb.run is not None and use_artifact:
        artifact = wandb.run.use_artifact(artifact)
    else:
        api = wandb.Api()
        artifact = api.artifact(artifact, type=artifact_type)
    save_dir = None if save_dir is None else os.fspath(save_dir)
    return artifact.download(root=save_dir)