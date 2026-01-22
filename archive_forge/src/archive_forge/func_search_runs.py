import os
from collections import OrderedDict
from itertools import zip_longest
from typing import List, Optional
from mlflow.entities import ExperimentTag, Metric, Param, RunStatus, RunTag, ViewType
from mlflow.entities.dataset_input import DatasetInput
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.tracking import GET_METRIC_HISTORY_MAX_RESULTS, SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking._tracking_service import utils
from mlflow.tracking.metric_value_conversion_utils import convert_metric_value_to_float_if_possible
from mlflow.utils import chunk_list
from mlflow.utils.async_logging.run_operations import RunOperations, get_combined_run_operations
from mlflow.utils.mlflow_tags import MLFLOW_USER
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import add_databricks_profile_info_to_artifact_uri
from mlflow.utils.validation import (
def search_runs(self, experiment_ids, filter_string='', run_view_type=ViewType.ACTIVE_ONLY, max_results=SEARCH_MAX_RESULTS_DEFAULT, order_by=None, page_token=None):
    """Search experiments that fit the search criteria.

        Args:
            experiment_ids: List of experiment IDs, or a single int or string id.
            filter_string: Filter query string, defaults to searching all runs.
            run_view_type: One of enum values ACTIVE_ONLY, DELETED_ONLY, or ALL runs
                defined in :py:class:`mlflow.entities.ViewType`.
            max_results: Maximum number of runs desired.
            order_by: List of columns to order by (e.g., "metrics.rmse"). The ``order_by`` column
                can contain an optional ``DESC`` or ``ASC`` value. The default is ``ASC``.
                The default ordering is to sort by ``start_time DESC``, then ``run_id``.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_runs`` call.

        Returns:
            A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`Run <mlflow.entities.Run>` objects that satisfy the search expressions.
            If the underlying tracking store supports pagination, the token for the next page may
            be obtained via the ``token`` attribute of the returned object.

        """
    if isinstance(experiment_ids, int) or is_string_type(experiment_ids):
        experiment_ids = [experiment_ids]
    return self.store.search_runs(experiment_ids=experiment_ids, filter_string=filter_string, run_view_type=run_view_type, max_results=max_results, order_by=order_by, page_token=page_token)