import base64
import json
import logging
import subprocess
import textwrap
import time
from typing import Any, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def run_creation(self) -> None:
    """Creates a Python file which will be deployed on beam."""
    script = textwrap.dedent('\n        import os\n        import transformers\n        from transformers import GPT2LMHeadModel, GPT2Tokenizer\n\n        model_name = "{model_name}"\n\n        def beam_langchain(**inputs):\n            prompt = inputs["prompt"]\n            length = inputs["max_length"]\n\n            tokenizer = GPT2Tokenizer.from_pretrained(model_name)\n            model = GPT2LMHeadModel.from_pretrained(model_name)\n            encodedPrompt = tokenizer.encode(prompt, return_tensors=\'pt\')\n            outputs = model.generate(encodedPrompt, max_length=int(length),\n              do_sample=True, pad_token_id=tokenizer.eos_token_id)\n            output = tokenizer.decode(outputs[0], skip_special_tokens=True)\n\n            print(output)  # noqa: T201\n            return {{"text": output}}\n\n        ')
    script_name = 'run.py'
    with open(script_name, 'w') as file:
        file.write(script.format(model_name=self.model_name))