import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
def get_file(self, filename) -> Optional[str]:
    if self.has_file(filename):
        return self._make_path(filename)
    else:
        return None