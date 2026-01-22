from typing import TYPE_CHECKING, Any, Callable, List, TypeVar
import ray
from ray.util.annotations import DeveloperAPI
def get_next_unordered(self, timeout=None, ignore_if_timedout=False):
    """Returns any of the next pending results.

        This returns some result produced by submit(), blocking for up to
        the specified timeout until it is available. Unlike get_next(), the
        results are not always returned in same order as submitted, which can
        improve performance.

        Returns:
            The next result.

        Raises:
            TimeoutError if the timeout is reached.

        Examples:
            .. testcode::

                import ray
                from ray.util.actor_pool import ActorPool

                @ray.remote
                class Actor:
                    def double(self, v):
                        return 2 * v

                a1, a2 = Actor.remote(), Actor.remote()
                pool = ActorPool([a1, a2])
                pool.submit(lambda a, v: a.double.remote(v), 1)
                pool.submit(lambda a, v: a.double.remote(v), 2)
                print(pool.get_next_unordered())
                print(pool.get_next_unordered())

            .. testoutput::
                :options: +MOCK

                4
                2
        """
    if not self.has_next():
        raise StopIteration('No more results to get')
    res, _ = ray.wait(list(self._future_to_actor), num_returns=1, timeout=timeout)
    timeout_msg = 'Timed out waiting for result'
    raise_timeout_after_ignore = False
    if res:
        [future] = res
    elif not ignore_if_timedout:
        raise TimeoutError(timeout_msg)
    else:
        raise_timeout_after_ignore = True
    i, a = self._future_to_actor.pop(future)
    self._return_actor(a)
    del self._index_to_future[i]
    self._next_return_index = max(self._next_return_index, i + 1)
    if raise_timeout_after_ignore:
        raise TimeoutError(timeout_msg + '. The task {} has been ignored.'.format(future))
    return ray.get(future)