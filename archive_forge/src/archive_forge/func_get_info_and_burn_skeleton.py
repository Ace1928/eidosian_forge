import sys
import os
import io
import pathlib
import re
import argparse
import zipfile
import json
import pickle
import pprint
import urllib.parse
from typing import (
import torch.utils.show_pickle
def get_info_and_burn_skeleton(path_or_bytesio, **kwargs):
    model_info = get_model_info(path_or_bytesio, **kwargs)
    skeleton = get_inline_skeleton()
    page = burn_in_info(skeleton, model_info)
    return page