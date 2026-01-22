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
def expand_group_to_two_letter_codes(self, grp_name):
    return [self.alpha3_to_alpha2.get(x, x) for x in GROUP_MEMBERS[grp_name][1]]