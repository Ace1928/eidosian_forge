import functools
import json
import logging
import os
import tempfile
from copy import deepcopy
from typing import Any, Dict, Optional
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.sklearn import _SklearnTrainingSession
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.mlflow_tags import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def log_feature_importance_plot(features, importance, importance_type):
    """
            Log feature importance plot.
            """
    import matplotlib.pyplot as plt
    indices = np.argsort(importance)
    features = np.array(features)[indices]
    importance = importance[indices]
    num_features = len(features)
    w, h = [6.4, 4.8]
    h = h + 0.1 * num_features if num_features > 10 else h
    fig, ax = plt.subplots(figsize=(w, h))
    yloc = np.arange(num_features)
    ax.barh(yloc, importance, align='center', height=0.5)
    ax.set_yticks(yloc)
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance ({importance_type})')
    fig.tight_layout()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            filepath = os.path.join(tmpdir, f'feature_importance_{imp_type}.png')
            fig.savefig(filepath)
            mlflow.log_artifact(filepath)
        finally:
            plt.close(fig)