import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_pip_requirements
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.types.llm import (
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import (
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def postprocess_output_for_llm_v1_embedding_task(self, input_prompts: Union[str, List[str]], output_tensers: List[List[int]]):
    """
        Wrap output data with usage information.

        Examples:
            .. code-block:: python
                input_prompt = ["hello world and hello mlflow"]
                output_embedding = [0.47137904, 0.4669448, ..., 0.69726706]
                output_dicts = postprocess_output_for_llm_v1_embedding_task(
                    input_prompt, output_embedding
                )
                assert output_dicts == [
                    {
                        "object": "list",
                        "data": [
                            {
                                "object": "embedding",
                                "index": 0,
                                "embedding": [0.47137904, 0.4669448, ..., 0.69726706],
                            }
                        ],
                        "usage": {"prompt_tokens": 8, "total_tokens": 8},
                    }
                ]

        Args:
            input_prompts: text input prompts
            output_tensers: List of output tensors that contain the generated embeddings

        Returns:
             Dictionaries containing the output embedding and usage information for each
             input prompt.
        """
    prompt_tokens = sum([len(self.model.tokenizer(prompt)['input_ids']) for prompt in input_prompts])
    return {'object': 'list', 'data': [{'object': 'embedding', 'index': i, 'embedding': tensor} for i, tensor in enumerate(output_tensers)], 'usage': {'prompt_tokens': prompt_tokens, 'total_tokens': prompt_tokens}}