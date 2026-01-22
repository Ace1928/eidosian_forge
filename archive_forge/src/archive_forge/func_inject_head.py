import os
from typing import TYPE_CHECKING, Sequence, Type, Union
from wandb.sdk.lib import filesystem, runid
from . import _dtypes
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia
def inject_head(self) -> None:
    join = ''
    if '<head>' in self.html:
        parts = self.html.split('<head>', 1)
        parts[0] = parts[0] + '<head>'
    elif '<html>' in self.html:
        parts = self.html.split('<html>', 1)
        parts[0] = parts[0] + '<html><head>'
        parts[1] = '</head>' + parts[1]
    else:
        parts = ['', self.html]
    parts.insert(1, '<base target="_blank"><link rel="stylesheet" type="text/css" href="https://app.wandb.ai/normalize.css" />')
    self.html = join.join(parts).strip()