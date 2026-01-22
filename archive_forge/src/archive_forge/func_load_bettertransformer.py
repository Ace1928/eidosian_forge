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
def load_bettertransformer(model, targeted_task, load_tokenizer=None, tokenizer=None, feature_extractor=None, load_feature_extractor=None, SUPPORTED_TASKS=None, subfolder: str='', token: Optional[Union[bool, str]]=None, revision: str='main', model_kwargs: Optional[Dict[str, Any]]=None, config: AutoConfig=None, hub_kwargs: Optional[Dict]=None, **kwargs):
    if model_kwargs is None:
        model_kwargs = {}
    if model is None:
        model_id = SUPPORTED_TASKS[targeted_task]['default']
    elif isinstance(model, str):
        model_id = model
    else:
        model_id = None
    model_classes = {'pt': SUPPORTED_TASKS[targeted_task]['pt']}
    framework, model = infer_framework_load_model(model, model_classes=model_classes, config=config, framework='pt', task=targeted_task, **hub_kwargs, **model_kwargs)
    if framework == 'tf':
        raise NotImplementedError('BetterTransormer is PyTorch-specific. It will not work with the provided TensorFlow model.')
    model = BetterTransformer.transform(model, **kwargs)
    return (model, model_id, tokenizer, feature_extractor)