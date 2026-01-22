from concurrent.futures import Future
from typing import Any, Callable, Optional
import pytest
import duet
@pytest.mark.skipif(grpc is None, reason='only run if grpc is installed')
def test_awaitable_grpc_future():

    class ConcreteGrpcFuture(grpc.Future):

        def cancel(self) -> bool:
            return True

        def cancelled(self) -> bool:
            return True

        def running(self) -> bool:
            return True

        def done(self) -> bool:
            return True

        def result(self, timeout: Optional[int]=None) -> Any:
            return 1234

        def exception(self, timeout=None) -> Optional[BaseException]:
            return None

        def add_done_callback(self, fn: Callable[[Any], Any]) -> None:
            pass

        def traceback(self, timeout=None):
            pass
    assert isinstance(duet.awaitable(ConcreteGrpcFuture()), duet.AwaitableFuture)