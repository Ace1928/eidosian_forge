import faulthandler
import json
import logging
import os
import time
import traceback
import warnings
from typing import Any, Dict, Optional
def override_error_code_in_rootcause_data(self, rootcause_error_file: str, rootcause_error: Dict[str, Any], error_code: int=0):
    """Modify the rootcause_error read from the file, to correctly set the exit code."""
    if 'message' not in rootcause_error:
        log.warning('child error file (%s) does not have field `message`. \ncannot override error code: %s', rootcause_error_file, error_code)
    elif isinstance(rootcause_error['message'], str):
        log.warning('child error file (%s) has a new message format. \nskipping error code override', rootcause_error_file)
    else:
        rootcause_error['message']['errorCode'] = error_code