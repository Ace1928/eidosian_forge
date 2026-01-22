import hashlib
from functools import cached_property
from inspect import isawaitable
from sentry_sdk import configure_scope, start_span
from sentry_sdk.consts import OP
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def on_operation(self):
    self._operation_name = self.execution_context.operation_name
    operation_type = 'query'
    op = OP.GRAPHQL_QUERY
    if self.execution_context.query.strip().startswith('mutation'):
        operation_type = 'mutation'
        op = OP.GRAPHQL_MUTATION
    elif self.execution_context.query.strip().startswith('subscription'):
        operation_type = 'subscription'
        op = OP.GRAPHQL_SUBSCRIPTION
    description = operation_type
    if self._operation_name:
        description += ' {}'.format(self._operation_name)
    Hub.current.add_breadcrumb(category='graphql.operation', data={'operation_name': self._operation_name, 'operation_type': operation_type})
    with configure_scope() as scope:
        if scope.span:
            self.graphql_span = scope.span.start_child(op=op, description=description)
        else:
            self.graphql_span = start_span(op=op, description=description)
    self.graphql_span.set_data('graphql.operation.type', operation_type)
    self.graphql_span.set_data('graphql.operation.name', self._operation_name)
    self.graphql_span.set_data('graphql.document', self.execution_context.query)
    self.graphql_span.set_data('graphql.resource_name', self._resource_name)
    yield
    self.graphql_span.finish()