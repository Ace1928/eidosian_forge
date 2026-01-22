from __future__ import annotations
import asyncio
from typing import Any, AsyncIterator, Dict, List, Literal, Union, cast
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
Callback handler that returns an async iterator.