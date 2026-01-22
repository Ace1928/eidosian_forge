import logging
import os
import sys
import json
def env_bool(key, default):
    if key in os.environ:
        return True if os.environ[key].lower() == 'true' or os.environ[key] == '1' else False
    return default