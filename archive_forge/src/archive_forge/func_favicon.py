from __future__ import annotations
from lazyops.types import BaseModel, lazyproperty
from fastapi.responses import HTMLResponse
from typing import Optional, Dict, Any
@property
def favicon(self) -> str:
    """Return favicon `<link>` tag, if applicable.
        Returns:
            A `<link>` tag if self.favicon_url is not empty, otherwise returns a placeholder meta tag.
        """
    return f"<link rel='icon' type='image/x-icon' href='{self.favicon_url}'>" if self.favicon_url else '<meta/>'