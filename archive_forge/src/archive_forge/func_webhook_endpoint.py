import atexit
import inspect
import os
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from .utils import experimental, is_gradio_available
from .utils._deprecation import _deprecate_method
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
@experimental
def webhook_endpoint(path: Optional[str]=None) -> Callable:
    """Decorator to start a [`WebhooksServer`] and register the decorated function as a webhook endpoint.

    This is a helper to get started quickly. If you need more flexibility (custom landing page or webhook secret),
    you can use [`WebhooksServer`] directly. You can register multiple webhook endpoints (to the same server) by using
    this decorator multiple times.

    Check out the [webhooks guide](../guides/webhooks_server) for a step-by-step tutorial on how to setup your
    server and deploy it on a Space.

    <Tip warning={true}>

    `webhook_endpoint` is experimental. Its API is subject to change in the future.

    </Tip>

    <Tip warning={true}>

    You must have `gradio` installed to use `webhook_endpoint` (`pip install --upgrade gradio`).

    </Tip>

    Args:
        path (`str`, optional):
            The URL path to register the webhook function. If not provided, the function name will be used as the path.
            In any case, all webhooks are registered under `/webhooks`.

    Examples:
        The default usage is to register a function as a webhook endpoint. The function name will be used as the path.
        The server will be started automatically at exit (i.e. at the end of the script).

        ```python
        from huggingface_hub import webhook_endpoint, WebhookPayload

        @webhook_endpoint
        async def trigger_training(payload: WebhookPayload):
            if payload.repo.type == "dataset" and payload.event.action == "update":
                # Trigger a training job if a dataset is updated
                ...

        # Server is automatically started at the end of the script.
        ```

        Advanced usage: register a function as a webhook endpoint and start the server manually. This is useful if you
        are running it in a notebook.

        ```python
        from huggingface_hub import webhook_endpoint, WebhookPayload

        @webhook_endpoint
        async def trigger_training(payload: WebhookPayload):
            if payload.repo.type == "dataset" and payload.event.action == "update":
                # Trigger a training job if a dataset is updated
                ...

        # Start the server manually
        trigger_training.run()
        ```
    """
    if callable(path):
        return webhook_endpoint()(path)

    @wraps(WebhooksServer.add_webhook)
    def _inner(func: Callable) -> Callable:
        app = _get_global_app()
        app.add_webhook(path)(func)
        if len(app.registered_webhooks) == 1:
            atexit.register(app.run)

        @wraps(app.run)
        def _run_now():
            atexit.unregister(app.run)
            app.run()
        func.run = _run_now
        return func
    return _inner