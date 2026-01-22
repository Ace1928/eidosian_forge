from __future__ import annotations
import asyncio
from typing import Optional, Dict, Any, List, Union, Type, TYPE_CHECKING
from .base import BaseGlobalClient
@property
def browser(self) -> 'SyncBrowser':
    """
        Returns the browser instance
        """
    if self._browser is None:
        self._browser = self.client.chromium.launch(headless=True, downloads_path=self.settings.module_path.joinpath('data').as_posix())
    return self._browser