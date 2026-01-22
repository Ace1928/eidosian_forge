import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
def to_model(self):
    blocks = self.blocks
    if len(blocks) > 0 and blocks[0] != P():
        blocks = [P()] + blocks
    if len(blocks) > 0 and blocks[-1] != P():
        blocks = blocks + [P()]
    if not blocks:
        blocks = [P(), P()]
    return internal.ReportViewspec(display_name=self.title, description=self.description, project=internal.Project(name=self.project, entity_name=self.entity), id=self.id, created_at=self._created_at, updated_at=self._updated_at, spec=internal.Spec(panel_settings=self._panel_settings, blocks=[b.to_model() for b in blocks], width=self.width, authors=self._authors, discussion_threads=self._discussion_threads, ref=self._ref))