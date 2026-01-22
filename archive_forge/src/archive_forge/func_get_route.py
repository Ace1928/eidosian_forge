from typing import Any, Dict, List, Optional
from mlflow.gateway.client import MlflowGatewayClient
from mlflow.gateway.config import LimitsConfig, Route
from mlflow.gateway.constants import MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE
from mlflow.gateway.utils import gateway_deprecated
from mlflow.utils import get_results_from_paginated_fn
@gateway_deprecated
def get_route(name: str) -> Route:
    """
    Retrieves a specific route from the MLflow Gateway service.

    This function creates an instance of MlflowGatewayClient and uses it to fetch a route by its
    name from the Gateway service.

    Args:
        name: The name of the route to fetch.

    Returns:
        An instance of the Route class representing the fetched route.

    """
    return MlflowGatewayClient().get_route(name)