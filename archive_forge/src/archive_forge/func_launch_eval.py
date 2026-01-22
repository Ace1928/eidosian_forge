import copy
import os
from pathlib import Path
from transformers import pipeline as _transformers_pipeline
from transformers.onnx import FeaturesManager
from transformers.onnx.utils import get_preprocessor
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from ...pipelines import ORT_SUPPORTED_TASKS
from ...pipelines import pipeline as _optimum_pipeline
from ...runs_base import Run, TimeBenchmark, get_autoclass_name, task_processing_map
from .. import ORTQuantizer
from ..configuration import QuantizationConfig
from ..modeling_ort import ORTModel
from ..preprocessors import QuantizationPreprocessor
from .calibrator import OnnxRuntimeCalibrator
from .utils import task_ortmodel_map
def launch_eval(self):
    kwargs = self.task_processor.get_pipeline_kwargs()
    ort_pipeline = _optimum_pipeline(task=self.task, model=self.ort_model, tokenizer=self.preprocessor, feature_extractor=self.preprocessor, accelerator='ort', **kwargs)
    transformers_pipeline = _transformers_pipeline(task=self.task, model=self.torch_model, tokenizer=self.preprocessor, feature_extractor=self.preprocessor, **kwargs)
    eval_dataset = self.get_eval_dataset()
    print('Running evaluation...')
    baseline_metrics_dict = self.task_processor.run_evaluation(eval_dataset, transformers_pipeline, self.metric_names)
    optimized_metrics_dict = self.task_processor.run_evaluation(eval_dataset, ort_pipeline, self.metric_names)
    baseline_metrics_dict.pop('total_time_in_seconds', None)
    baseline_metrics_dict.pop('samples_per_second', None)
    baseline_metrics_dict.pop('latency_in_seconds', None)
    optimized_metrics_dict.pop('total_time_in_seconds', None)
    optimized_metrics_dict.pop('samples_per_second', None)
    optimized_metrics_dict.pop('latency_in_seconds', None)
    self.return_body['evaluation']['others']['baseline'].update(baseline_metrics_dict)
    self.return_body['evaluation']['others']['optimized'].update(optimized_metrics_dict)