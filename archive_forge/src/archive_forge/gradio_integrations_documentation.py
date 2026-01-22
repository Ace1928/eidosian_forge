import logging
from typing import Callable
from ray import serve
from ray.serve._private.http_util import ASGIAppReplicaWrapper
from ray.util.annotations import PublicAPI
Builds and wraps an ASGI app from the provided builder.

        The builder should take no arguments and return a Gradio App (of type Interface
        or Blocks).
        