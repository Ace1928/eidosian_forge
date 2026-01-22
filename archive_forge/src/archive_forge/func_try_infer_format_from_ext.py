from argparse import ArgumentParser
from ..pipelines import Pipeline, PipelineDataFormat, get_supported_tasks, pipeline
from ..utils import logging
from . import BaseTransformersCLICommand
def try_infer_format_from_ext(path: str):
    if not path:
        return 'pipe'
    for ext in PipelineDataFormat.SUPPORTED_FORMATS:
        if path.endswith(ext):
            return ext
    raise Exception(f'Unable to determine file format from file extension {path}. Please provide the format through --format {PipelineDataFormat.SUPPORTED_FORMATS}')