import json
import os
from docker import errors
from docker.context.config import get_meta_dir
from docker.context.config import METAFILE
from docker.context.config import get_current_context_name
from docker.context.config import write_context_name_to_docker_config
from docker.context import Context
@classmethod
def get_current_context(cls):
    """Get current context.
        Returns:
            (Context): current context object.
        """
    return cls.get_context()