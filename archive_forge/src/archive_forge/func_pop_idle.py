from typing import TYPE_CHECKING, Any, Callable, List, TypeVar
import ray
from ray.util.annotations import DeveloperAPI
def pop_idle(self):
    """Removes an idle actor from the pool.

        Returns:
            An idle actor if one is available.
            None if no actor was free to be removed.

        Examples:
            .. testcode::

                import ray
                from ray.util.actor_pool import ActorPool

                @ray.remote
                class Actor:
                    def double(self, v):
                        return 2 * v

                a1 = Actor.remote()
                pool = ActorPool([a1])
                pool.submit(lambda a, v: a.double.remote(v), 1)
                assert pool.pop_idle() is None
                assert pool.get_next() == 2
                assert pool.pop_idle() == a1

        """
    if self.has_free():
        return self._idle_actors.pop()
    return None