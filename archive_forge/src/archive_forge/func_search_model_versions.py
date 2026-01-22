from typing import Any, Dict, List, Optional
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.client import MlflowClient
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.logging_utils import eprint
def search_model_versions(max_results: Optional[int]=None, filter_string: Optional[str]=None, order_by: Optional[List[str]]=None) -> List[ModelVersion]:
    """Search for model versions that satisfy the filter criteria.

    .. warning:

        The model version search results may not have aliases populated for performance reasons.

    Args:
        filter_string: Filter query string
            (e.g., ``"name = 'a_model_name' and tag.key = 'value1'"``),
            defaults to searching for all model versions. The following identifiers, comparators,
            and logical operators are supported.

            Identifiers
              - ``name``: model name.
              - ``source_path``: model version source path.
              - ``run_id``: The id of the mlflow run that generates the model version.
              - ``tags.<tag_key>``: model version tag. If ``tag_key`` contains spaces, it must be
                wrapped with backticks (e.g., ``"tags.`extra key`"``).

            Comparators
              - ``=``: Equal to.
              - ``!=``: Not equal to.
              - ``LIKE``: Case-sensitive pattern match.
              - ``ILIKE``: Case-insensitive pattern match.
              - ``IN``: In a value list. Only ``run_id`` identifier supports ``IN`` comparator.

            Logical operators
              - ``AND``: Combines two sub-queries and returns True if both of them are True.

        max_results: If passed, specifies the maximum number of models desired. If not
            passed, all models will be returned.
        order_by: List of column names with ASC|DESC annotation, to be used for ordering
            matching search results.

    Returns:
        A list of :py:class:`mlflow.entities.model_registry.ModelVersion` objects
            that satisfy the search expressions.

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow
        from sklearn.linear_model import LogisticRegression

        for _ in range(2):
            with mlflow.start_run():
                mlflow.sklearn.log_model(
                    LogisticRegression(),
                    "Cordoba",
                    registered_model_name="CordobaWeatherForecastModel",
                )

        # Get all versions of the model filtered by name
        filter_string = "name = 'CordobaWeatherForecastModel'"
        results = mlflow.search_model_versions(filter_string=filter_string)
        print("-" * 80)
        for res in results:
            print(f"name={res.name}; run_id={res.run_id}; version={res.version}")

        # Get the version of the model filtered by run_id
        filter_string = "run_id = 'ae9a606a12834c04a8ef1006d0cff779'"
        results = mlflow.search_model_versions(filter_string=filter_string)
        print("-" * 80)
        for res in results:
            print(f"name={res.name}; run_id={res.run_id}; version={res.version}")

    .. code-block:: text
        :caption: Output

        --------------------------------------------------------------------------------
        name=CordobaWeatherForecastModel; run_id=ae9a606a12834c04a8ef1006d0cff779; version=2
        name=CordobaWeatherForecastModel; run_id=d8f028b5fedf4faf8e458f7693dfa7ce; version=1
        --------------------------------------------------------------------------------
        name=CordobaWeatherForecastModel; run_id=ae9a606a12834c04a8ef1006d0cff779; version=2
    """

    def pagination_wrapper_func(number_to_get, next_page_token):
        return MlflowClient().search_model_versions(max_results=number_to_get, filter_string=filter_string, order_by=order_by, page_token=next_page_token)
    return get_results_from_paginated_fn(paginated_fn=pagination_wrapper_func, max_results_per_page=SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT, max_results=max_results)