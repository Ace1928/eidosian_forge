import re
import os.path
from typing import Dict, Union, Optional
from os.path import join as pjoin
def get_pricing_file_path(file_path=None):
    if os.path.exists(CUSTOM_PRICING_FILE_PATH) and os.path.isfile(CUSTOM_PRICING_FILE_PATH):
        return CUSTOM_PRICING_FILE_PATH
    return DEFAULT_PRICING_FILE_PATH