from typing import Any, Dict, Optional, Union
from transformers import (
from transformers import pipeline as transformers_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.onnx.utils import get_preprocessor
from transformers.pipelines import SUPPORTED_TASKS as TRANSFORMERS_SUPPORTED_TASKS
from transformers.pipelines import infer_framework_load_model
from ..bettertransformer import BetterTransformer
from ..utils import is_onnxruntime_available
from ..utils.file_utils import find_files_matching_pattern
Pipelines running different backends.