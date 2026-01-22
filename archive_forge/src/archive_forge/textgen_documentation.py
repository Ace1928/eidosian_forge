import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
import requests
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Field
Yields results objects as they are generated in real time.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.

        Args:
            prompt: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens being generated.

        Yields:
            A dictionary like objects containing a string token and metadata.
            See text-generation-webui docs and below for more.

        Example:
            .. code-block:: python

                from langchain_community.llms import TextGen
                llm = TextGen(
                    model_url = "ws://localhost:5005"
                    streaming=True
                )
                for chunk in llm.stream("Ask 'Hi, how are you?' like a pirate:'",
                        stop=["'","
"]):
                    print(chunk, end='', flush=True)  # noqa: T201

        