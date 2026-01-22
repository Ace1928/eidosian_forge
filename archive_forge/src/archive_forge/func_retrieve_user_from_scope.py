from typing import TYPE_CHECKING
from pydantic import BaseModel  # type: ignore
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.utils import event_from_exception, transaction_from_function
def retrieve_user_from_scope(scope: 'Scope') -> 'Optional[Dict[str, Any]]':
    scope_user = scope.get('user', {})
    if not scope_user:
        return None
    if isinstance(scope_user, dict):
        return scope_user
    if isinstance(scope_user, BaseModel):
        return scope_user.dict()
    if hasattr(scope_user, 'asdict'):
        return scope_user.asdict()
    plugin = get_plugin_for_value(scope_user)
    if plugin and (not is_async_callable(plugin.to_dict)):
        return plugin.to_dict(scope_user)
    return None