from typing import Any, Dict, List, Mapping, Optional, Union
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, SecretStr

            Attempt to retrieve the fine-tuned adapter from a Predibase repository.
            If absent, then load the fine-tuned adapter from a HuggingFace repository.
            