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
def resolve_lang_code(self, src, tgt) -> Tuple[str, str]:
    src_tags = self.get_tags(src, self.tag2name[src])
    tgt_tags = self.get_tags(tgt, self.tag2name[tgt])
    return (src_tags, tgt_tags)