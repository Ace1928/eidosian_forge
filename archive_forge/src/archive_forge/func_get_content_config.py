from __future__ import annotations
import os
import pickle
import typing as t
from .constants import (
from .compat.packaging import (
from .compat.yaml import (
from .io import (
from .util import (
from .data import (
from .config import (
def get_content_config(args: EnvironmentConfig) -> ContentConfig:
    """
    Parse and return the content configuration (if any) for the current collection.
    For ansible-core, a default configuration is used.
    Results are cached.
    """
    if args.host_path:
        args.content_config = deserialize_content_config(os.path.join(args.host_path, 'config.dat'))
    if args.content_config:
        return args.content_config
    collection_config_path = 'tests/config.yml'
    config = None
    if data_context().content.collection and os.path.exists(collection_config_path):
        config = load_config(collection_config_path)
    if not config:
        config = parse_content_config(dict(modules=dict(python_requires='default')))
    if not config.modules.python_versions:
        raise ApplicationError('This collection does not declare support for modules/module_utils on any known Python version.\nAnsible supports modules/module_utils on Python versions: %s\nThis collection provides the Python requirement: %s' % (', '.join(SUPPORTED_PYTHON_VERSIONS), config.modules.python_requires))
    args.content_config = config
    return config