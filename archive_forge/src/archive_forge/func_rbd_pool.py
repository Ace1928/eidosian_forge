from __future__ import annotations
import io
from typing import NoReturn, Optional  # noqa: H301
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick import utils
@property
def rbd_pool(self) -> str:
    return self._rbd_volume.pool