from __future__ import annotations
from io import IOBase
from lazyops.types import BaseModel, Field
from lazyops.utils import logger
from typing import Optional, Dict, Any, List, Union, Sequence, Callable, TYPE_CHECKING
from .types import SlackContext, SlackPayload
from .configs import SlackSettings
def create_slash_command_endpoint(self, router: Union['APIRouter', 'FastAPI'], path: str, function: Callable, function_name: Optional[str]=None):
    function_name = function_name or function.__qualname__
    from fastapi.requests import Request

    async def slash_command(data: Request):
        payload = await data.form()
        return await function(SlackPayload(**payload))
    router.add_api_route(path=path, endpoint=slash_command, name=function_name, methods=['POST'], include_in_schema=False)
    return router