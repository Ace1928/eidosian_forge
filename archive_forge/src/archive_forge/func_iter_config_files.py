import configparser
import locale
import os
import sys
from typing import Any, Dict, Iterable, List, NewType, Optional, Tuple
from pip._internal.exceptions import (
from pip._internal.utils import appdirs
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import ensure_dir, enum
def iter_config_files(self) -> Iterable[Tuple[Kind, List[str]]]:
    """Yields variant and configuration files associated with it.

        This should be treated like items of a dictionary. The order
        here doesn't affect what gets overridden. That is controlled
        by OVERRIDE_ORDER. However this does control the order they are
        displayed to the user. It's probably most ergononmic to display
        things in the same order as OVERRIDE_ORDER
        """
    env_config_file = os.environ.get('PIP_CONFIG_FILE', None)
    config_files = get_configuration_files()
    yield (kinds.GLOBAL, config_files[kinds.GLOBAL])
    should_load_user_config = not self.isolated and (not (env_config_file and os.path.exists(env_config_file)))
    if should_load_user_config:
        yield (kinds.USER, config_files[kinds.USER])
    yield (kinds.SITE, config_files[kinds.SITE])
    if env_config_file is not None:
        yield (kinds.ENV, [env_config_file])
    else:
        yield (kinds.ENV, [])