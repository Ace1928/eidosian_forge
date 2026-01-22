import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
import yaml
from ..models import auto as auto_module
from ..models.auto.configuration_auto import model_type_to_module_name
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand
def disable_fx_test(filename: Path) -> bool:
    with open(filename) as fp:
        content = fp.read()
    new_content = re.sub('fx_compatible\\s*=\\s*True', 'fx_compatible = False', content)
    with open(filename, 'w') as fp:
        fp.write(new_content)
    return content != new_content