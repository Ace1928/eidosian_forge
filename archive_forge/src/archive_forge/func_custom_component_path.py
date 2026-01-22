from __future__ import annotations
import asyncio
import contextlib
import sys
import inspect
import json
import mimetypes
import os
import posixpath
import secrets
import threading
import time
import traceback
from pathlib import Path
from queue import Empty as EmptyQueue
from typing import (
import fastapi
import httpx
import markupsafe
import orjson
from fastapi import (
from fastapi.responses import (
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio_client.utils import ServerMessage
from jinja2.exceptions import TemplateNotFound
from multipart.multipart import parse_options_header
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.responses import RedirectResponse, StreamingResponse
import gradio
from gradio import ranged_response, route_utils, utils, wasm_utils
from gradio.context import Context
from gradio.data_classes import (
from gradio.exceptions import Error
from gradio.oauth import attach_oauth
from gradio.route_utils import (  # noqa: F401
from gradio.server_messages import (
from gradio.state_holder import StateHolder
from gradio.utils import get_package_version, get_upload_folder
@app.get('/custom_component/{id}/{type}/{file_name}')
def custom_component_path(id: str, type: str, file_name: str):
    config = app.get_blocks().config
    components = config['components']
    location = next((item for item in components if item['component_class_id'] == id), None)
    if location is None:
        raise HTTPException(status_code=404, detail='Component not found.')
    component_instance = app.get_blocks().get_component(location['id'])
    module_name = component_instance.__class__.__module__
    module_path = sys.modules[module_name].__file__
    if module_path is None or component_instance is None:
        raise HTTPException(status_code=404, detail='Component not found.')
    return FileResponse(safe_join(str(Path(module_path).parent), f'{component_instance.__class__.TEMPLATE_DIR}/{type}/{file_name}'))