import argparse
import datetime
import json
import os
import re
from pathlib import Path
from typing import Tuple
import yaml
from tqdm import tqdm
from transformers.models.marian.convert_marian_to_pytorch import (
def download_lang_info(self):
    Path(LANG_CODE_PATH).parent.mkdir(exist_ok=True)
    import wget
    if not os.path.exists(ISO_PATH):
        wget.download(ISO_URL, ISO_PATH)
    if not os.path.exists(LANG_CODE_PATH):
        wget.download(LANG_CODE_URL, LANG_CODE_PATH)