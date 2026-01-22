import warnings
from argparse import ArgumentParser
from os import listdir, makedirs
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from packaging.version import Version, parse
from transformers.pipelines import Pipeline, pipeline
from transformers.tokenization_utils import BatchEncoding
from transformers.utils import ModelOutput, is_tf_available, is_torch_available
def load_graph_from_args(pipeline_name: str, framework: str, model: str, tokenizer: Optional[str]=None, **models_kwargs) -> Pipeline:
    """
    Convert the set of arguments provided through the CLI to an actual pipeline reference (tokenizer + model

    Args:
        pipeline_name: The kind of pipeline to use (ner, question-answering, etc.)
        framework: The actual model to convert the pipeline from ("pt" or "tf")
        model: The model name which will be loaded by the pipeline
        tokenizer: The tokenizer name which will be loaded by the pipeline, default to the model's value

    Returns: Pipeline object

    """
    if tokenizer is None:
        tokenizer = model
    if framework == 'pt' and (not is_torch_available()):
        raise Exception('Cannot convert because PyTorch is not installed. Please install torch first.')
    if framework == 'tf' and (not is_tf_available()):
        raise Exception('Cannot convert because TF is not installed. Please install tensorflow first.')
    print(f'Loading pipeline (model: {model}, tokenizer: {tokenizer})')
    return pipeline(pipeline_name, model=model, tokenizer=tokenizer, framework=framework, model_kwargs=models_kwargs)