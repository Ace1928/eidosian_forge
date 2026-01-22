from __future__ import annotations
import logging
import os.path
import re
from dataclasses import dataclass, field
from typing import Optional
from watchdog.utils.patterns import match_any_paths
def on_closed(self, event: FileSystemEvent) -> None:
    super().on_closed(event)
    self.logger.info('Closed file: %s', event.src_path)