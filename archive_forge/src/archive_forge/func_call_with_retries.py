import enum
import functools
import random
import threading
import time
import grpc
from tensorboard import version
from tensorboard.util import tb_logging
def call_with_retries(api_method, request, clock=None):
    """Call a gRPC stub API method, with automatic retry logic.

    This only supports unary-unary RPCs: i.e., no streaming on either end.
    Streamed RPCs will generally need application-level pagination support,
    because after a gRPC error one must retry the entire request; there is no
    "retry-resume" functionality.

    Retries are handled with jittered exponential backoff to spread out failures
    due to request spikes.

    Args:
      api_method: Callable for the API method to invoke.
      request: Request protocol buffer to pass to the API method.
      clock: an interface object supporting `time()` and `sleep()` methods
        like the standard `time` module; if not passed, uses the normal module.

    Returns:
      Response protocol buffer returned by the API method.

    Raises:
      grpc.RpcError: if a non-retryable error is returned, or if all retry
        attempts have been exhausted.
    """
    if clock is None:
        clock = time
    rpc_name = request.__class__.__name__.replace('Request', '')
    logger.debug('RPC call %s with request: %r', rpc_name, request)
    num_attempts = 0
    while True:
        num_attempts += 1
        try:
            return api_method(request, timeout=_GRPC_DEFAULT_TIMEOUT_SECS, metadata=version_metadata())
        except grpc.RpcError as e:
            logger.info('RPC call %s got error %s', rpc_name, e)
            if e.code() not in _GRPC_RETRYABLE_STATUS_CODES:
                raise
            if num_attempts >= _GRPC_RETRY_MAX_ATTEMPTS:
                raise
        backoff_secs = _compute_backoff_seconds(num_attempts)
        logger.info('RPC call %s attempted %d times, retrying in %.1f seconds', rpc_name, num_attempts, backoff_secs)
        clock.sleep(backoff_secs)