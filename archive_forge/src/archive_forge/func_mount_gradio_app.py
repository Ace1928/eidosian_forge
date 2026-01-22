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
@document()
def mount_gradio_app(app: fastapi.FastAPI, blocks: gradio.Blocks, path: str, app_kwargs: dict[str, Any] | None=None, *, auth: Callable | tuple[str, str] | list[tuple[str, str]] | None=None, auth_message: str | None=None, auth_dependency: Callable[[fastapi.Request], str | None] | None=None, root_path: str | None=None, allowed_paths: list[str] | None=None, blocked_paths: list[str] | None=None, favicon_path: str | None=None, show_error: bool=True) -> fastapi.FastAPI:
    """Mount a gradio.Blocks to an existing FastAPI application.

    Parameters:
        app: The parent FastAPI application.
        blocks: The blocks object we want to mount to the parent app.
        path: The path at which the gradio application will be mounted.
        app_kwargs: Additional keyword arguments to pass to the underlying FastAPI app as a dictionary of parameter keys and argument values. For example, `{"docs_url": "/docs"}`
        auth: If provided, username and password (or list of username-password tuples) required to access the gradio app. Can also provide function that takes username and password and returns True if valid login.
        auth_message: If provided, HTML message provided on login page for this gradio app.
        auth_dependency: A function that takes a FastAPI request and returns a string user ID or None. If the function returns None for a specific request, that user is not authorized to access the gradio app (they will see a 401 Unauthorized response). To be used with external authentication systems like OAuth. Cannot be used with `auth`.
        root_path: The subpath corresponding to the public deployment of this FastAPI application. For example, if the application is served at "https://example.com/myapp", the `root_path` should be set to "/myapp". A full URL beginning with http:// or https:// can be provided, which will be used in its entirety. Normally, this does not need to provided (even if you are using a custom `path`). However, if you are serving the FastAPI app behind a proxy, the proxy may not provide the full path to the Gradio app in the request headers. In which case, you can provide the root path here.
        allowed_paths: List of complete filepaths or parent directories that this gradio app is allowed to serve. Must be absolute paths. Warning: if you provide directories, any files in these directories or their subdirectories are accessible to all users of your app.
        blocked_paths: List of complete filepaths or parent directories that this gradio app is not allowed to serve (i.e. users of your app are not allowed to access). Must be absolute paths. Warning: takes precedence over `allowed_paths` and all other directories exposed by Gradio by default.
        favicon_path: If a path to a file (.png, .gif, or .ico) is provided, it will be used as the favicon for this gradio app's page.
        show_error: If True, any errors in the gradio app will be displayed in an alert modal and printed in the browser console log. Otherwise, errors will only be visible in the terminal session running the Gradio app.
    Example:
        from fastapi import FastAPI
        import gradio as gr
        app = FastAPI()
        @app.get("/")
        def read_main():
            return {"message": "This is your main app"}
        io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
        app = gr.mount_gradio_app(app, io, path="/gradio")
        # Then run `uvicorn run:app` from the terminal and navigate to http://localhost:8000/gradio.
    """
    blocks.dev_mode = False
    blocks.config = blocks.get_config_file()
    blocks.validate_queue_settings()
    if auth is not None and auth_dependency is not None:
        raise ValueError('You cannot provide both `auth` and `auth_dependency` in mount_gradio_app(). Please choose one.')
    if auth and (not callable(auth)) and (not isinstance(auth[0], tuple)) and (not isinstance(auth[0], list)):
        blocks.auth = [auth]
    else:
        blocks.auth = auth
    blocks.auth_message = auth_message
    blocks.favicon_path = favicon_path
    blocks.allowed_paths = allowed_paths or []
    blocks.blocked_paths = blocked_paths or []
    blocks.show_error = show_error
    if not isinstance(blocks.allowed_paths, list):
        raise ValueError('`allowed_paths` must be a list of directories.')
    if not isinstance(blocks.blocked_paths, list):
        raise ValueError('`blocked_paths` must be a list of directories.')
    if root_path is not None:
        blocks.root_path = root_path
    gradio_app = App.create_app(blocks, app_kwargs=app_kwargs, auth_dependency=auth_dependency)
    old_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def new_lifespan(app: FastAPI):
        async with old_lifespan(app):
            async with gradio_app.router.lifespan_context(gradio_app):
                gradio_app.get_blocks().startup_events()
                yield
    app.router.lifespan_context = new_lifespan
    app.mount(path, gradio_app)
    return app