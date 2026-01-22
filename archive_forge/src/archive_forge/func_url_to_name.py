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
def url_to_name(url):
    return url.split('/')[-1].split('.')[0]