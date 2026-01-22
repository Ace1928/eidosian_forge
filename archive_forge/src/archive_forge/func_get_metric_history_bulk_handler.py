import bisect
import json
import logging
import os
import pathlib
import posixpath
import re
import tempfile
import time
import urllib
from functools import wraps
from typing import List, Set
import requests
from flask import Response, current_app, jsonify, request, send_file
from google.protobuf import descriptor
from google.protobuf.json_format import ParseError
from mlflow.entities import DatasetInput, ExperimentTag, FileInfo, Metric, Param, RunTag, ViewType
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.multipart_upload import MultipartUploadPart
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_TARGET
from mlflow.exceptions import MlflowException, _UnsupportedMultipartUploadException
from mlflow.models import Model
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import (
from mlflow.protos.mlflow_artifacts_pb2 import (
from mlflow.protos.mlflow_artifacts_pb2 import (
from mlflow.protos.model_registry_pb2 import (
from mlflow.protos.service_pb2 import (
from mlflow.server.validation import _validate_content_type
from mlflow.store.artifact.artifact_repo import MultipartUploadMixin
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.tracking._model_registry import utils as registry_utils
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service import utils
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mime_type_utils import _guess_mime_type
from mlflow.utils.promptlab_utils import _create_promptlab_run_impl
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.uri import is_local_uri, validate_path_is_safe, validate_query_string
from mlflow.utils.validation import _validate_batch_log_api_req
@catch_mlflow_exception
@_disable_if_artifacts_only
def get_metric_history_bulk_handler():
    MAX_HISTORY_RESULTS = 25000
    MAX_RUN_IDS_PER_REQUEST = 100
    run_ids = request.args.to_dict(flat=False).get('run_id', [])
    if not run_ids:
        raise MlflowException(message='GetMetricHistoryBulk request must specify at least one run_id.', error_code=INVALID_PARAMETER_VALUE)
    if len(run_ids) > MAX_RUN_IDS_PER_REQUEST:
        raise MlflowException(message=f'GetMetricHistoryBulk request cannot specify more than {MAX_RUN_IDS_PER_REQUEST} run_ids. Received {len(run_ids)} run_ids.', error_code=INVALID_PARAMETER_VALUE)
    metric_key = request.args.get('metric_key')
    if metric_key is None:
        raise MlflowException(message='GetMetricHistoryBulk request must specify a metric_key.', error_code=INVALID_PARAMETER_VALUE)
    max_results = int(request.args.get('max_results', MAX_HISTORY_RESULTS))
    max_results = min(max_results, MAX_HISTORY_RESULTS)
    store = _get_tracking_store()

    def _default_history_bulk_impl():
        metrics_with_run_ids = []
        for run_id in sorted(run_ids):
            metrics_for_run = sorted(store.get_metric_history(run_id=run_id, metric_key=metric_key, max_results=max_results), key=lambda metric: (metric.timestamp, metric.step, metric.value))
            metrics_with_run_ids.extend([{'key': metric.key, 'value': metric.value, 'timestamp': metric.timestamp, 'step': metric.step, 'run_id': run_id} for metric in metrics_for_run])
        return metrics_with_run_ids
    if hasattr(store, 'get_metric_history_bulk'):
        metrics_with_run_ids = [metric.to_dict() for metric in store.get_metric_history_bulk(run_ids=run_ids, metric_key=metric_key, max_results=max_results)]
    else:
        metrics_with_run_ids = _default_history_bulk_impl()
    return {'metrics': metrics_with_run_ids[:max_results]}