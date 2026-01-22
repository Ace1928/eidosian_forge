import asyncio
import logging
import time
from abc import ABC, abstractmethod
from asyncio.tasks import FIRST_COMPLETED
from typing import Any, Callable, Optional, Union
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import calculate_remaining_timeout
from ray.serve.handle import DeploymentResponse, DeploymentResponseGenerator
def cancelled(self) -> bool:
    return self._response.cancelled()