import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def load_general_config(config_path: Optional[str]=None) -> Dict:
    config_file = find_config_file(config_path)
    if not config_file:
        return {}
    try:
        with open(config_file) as f:
            conf: Dict = json.load(f)
            return conf
    except (OSError, ValueError) as e:
        log.debug(e)
    log.debug('All parsing attempts failed - returning empty config')
    return {}