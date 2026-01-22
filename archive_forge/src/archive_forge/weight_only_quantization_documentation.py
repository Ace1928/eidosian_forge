import importlib
from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra
from langchain_community.llms.utils import enforce_stop_tokens
Call the HuggingFace model and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The generated text.

        Example:
            .. code-block:: python

                from langchain_community.llms import WeightOnlyQuantPipeline
                llm = WeightOnlyQuantPipeline.from_model_id(
                    model_id="google/flan-t5-large",
                    task="text2text-generation",
                )
                llm("This is a prompt.")
        