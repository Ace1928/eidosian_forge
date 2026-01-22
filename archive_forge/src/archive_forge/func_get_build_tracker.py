import contextlib
import hashlib
import logging
import os
from types import TracebackType
from typing import Dict, Generator, Optional, Set, Type, Union
from pip._internal.models.link import Link
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.temp_dir import TempDirectory
@contextlib.contextmanager
def get_build_tracker() -> Generator['BuildTracker', None, None]:
    root = os.environ.get('PIP_BUILD_TRACKER')
    with contextlib.ExitStack() as ctx:
        if root is None:
            root = ctx.enter_context(TempDirectory(kind='build-tracker')).path
            ctx.enter_context(update_env_context_manager(PIP_BUILD_TRACKER=root))
            logger.debug('Initialized build tracking at %s', root)
        with BuildTracker(root) as tracker:
            yield tracker