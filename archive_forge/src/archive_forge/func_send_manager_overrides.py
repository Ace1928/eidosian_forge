import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Optional
@property
def send_manager_overrides(self):
    overrides = {}
    if self.entity:
        overrides['entity'] = self.entity
    if self.project:
        overrides['project'] = self.project
    return overrides