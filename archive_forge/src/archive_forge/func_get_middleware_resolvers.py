import inspect
from functools import partial
from itertools import chain
from wandb_promise import Promise
def get_middleware_resolvers(middlewares):
    for middleware in middlewares:
        if inspect.isfunction(middleware):
            yield middleware
        if not hasattr(middleware, MIDDLEWARE_RESOLVER_FUNCTION):
            continue
        yield getattr(middleware, MIDDLEWARE_RESOLVER_FUNCTION)