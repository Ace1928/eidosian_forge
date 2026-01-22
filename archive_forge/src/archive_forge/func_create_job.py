import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import wandb
from wandb.apis.internal import Api
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.internal.job_builder import JobBuilder
from wandb.sdk.launch.builder.build import get_current_python_version
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.utils import _is_git_uri
from wandb.sdk.lib import filesystem
from wandb.util import make_artifact_name_safe
def create_job(path: str, job_type: str, entity: Optional[str]=None, project: Optional[str]=None, name: Optional[str]=None, description: Optional[str]=None, aliases: Optional[List[str]]=None, runtime: Optional[str]=None, entrypoint: Optional[str]=None, git_hash: Optional[str]=None) -> Optional[Artifact]:
    """Create a job from a path, not as the output of a run.

    Arguments:
        path (str): Path to the job directory.
        job_type (str): Type of the job. One of "git", "code", or "image".
        entity (Optional[str]): Entity to create the job under.
        project (Optional[str]): Project to create the job under.
        name (Optional[str]): Name of the job.
        description (Optional[str]): Description of the job.
        aliases (Optional[List[str]]): Aliases for the job.
        runtime (Optional[str]): Python runtime of the job, like 3.9.
        entrypoint (Optional[str]): Entrypoint of the job.
        git_hash (Optional[str]): Git hash of a specific commit, when using git type jobs.


    Returns:
        Optional[Artifact]: The artifact created by the job, the action (for printing), and job aliases.
                            None if job creation failed.

    Example:
        ```python
        artifact_job = wandb.create_job(
            job_type="code",
            path=".",
            entity="wandb",
            project="jobs",
            name="my-train-job",
            description="My training job",
            aliases=["train"],
            runtime="3.9",
            entrypoint="train.py",
        )
        # then run the newly created job
        artifact_job.call()
        ```
    """
    api = Api()
    artifact_job, _action, _aliases = _create_job(api, job_type, path, entity, project, name, description, aliases, runtime, entrypoint, git_hash)
    return artifact_job