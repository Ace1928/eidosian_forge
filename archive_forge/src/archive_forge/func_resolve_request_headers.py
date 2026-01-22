import logging
import warnings
import entrypoints
from mlflow.tracking.request_header.databricks_request_header_provider import (
from mlflow.tracking.request_header.default_request_header_provider import (
def resolve_request_headers(request_headers=None):
    """Generate a set of request headers from registered providers.

    Request headers are resolved in the order that providers are registered. Argument headers are
    applied last. This function iterates through all request header providers in the registry.
    Additional context providers can be registered as described in
    :py:class:`mlflow.tracking.request_header.RequestHeaderProvider`.

    Args:
        tags: A dictionary of request headers to override. If specified, headers passed in this
            argument will override those inferred from the context.

    Returns:
        A dictionary of resolved headers.
    """
    all_request_headers = {}
    for provider in _request_header_provider_registry:
        try:
            if provider.in_context():
                for header, value in provider.request_headers().items():
                    all_request_headers[header] = f'{all_request_headers[header]} {value}' if header in all_request_headers else value
        except Exception as e:
            _logger.warning('Encountered unexpected error during resolving request headers: %s', e)
    if request_headers is not None:
        all_request_headers.update(request_headers)
    return all_request_headers