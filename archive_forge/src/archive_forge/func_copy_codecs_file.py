from abc import ABC, abstractmethod
from functools import lru_cache
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from typing_extensions import final
from parlai.core.build_data import download, make_dir
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
from parlai.utils.typing import TShared
import parlai.utils.logging as logging
def copy_codecs_file(self, target_file: str):
    """
        Copy the codecs file to a new location.

        :param target_file:
            where to copy the codecs.
        """
    with open(target_file, 'w', encoding='utf-8') as wfile:
        with open(self.codecs, encoding='utf-8') as rfile:
            for line in rfile:
                wfile.write(line)