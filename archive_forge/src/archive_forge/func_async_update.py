from collections import defaultdict, deque
from functools import partial
import pathlib
from typing import (
import uuid
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.learner.learner import LearnerSpec
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.minibatch_utils import ShardBatchIterator
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.train._internal.backend_executor import BackendExecutor
from ray.tune.utils.file_transfer import sync_dir_between_nodes
from ray.util.annotations import PublicAPI
def async_update(self, batch: MultiAgentBatch, *, minibatch_size: Optional[int]=None, num_iters: int=1, reduce_fn: Optional[Callable[[List[Mapping[str, Any]]], ResultDict]]=_reduce_mean_results) -> Union[List[Mapping[str, Any]], List[List[Mapping[str, Any]]]]:
    """Asnychronously do gradient based updates to the Learner(s) with `batch`.

        Args:
            batch: The data batch to use for the update.
            minibatch_size: The minibatch size to use for the update.
            num_iters: The number of complete passes over all the sub-batches in the
                input multi-agent batch.
            reduce_fn: An optional callable to reduce the results from a list of the
                Learner actors into a single result. This can be any arbitrary function
                that takes a list of dictionaries and returns a single dictionary. For
                example you can either take an average (default) or concatenate the
                results (for example for metrics) or be more selective about you want to
                report back to the algorithm's training_step. If None is passed, the
                results will not get reduced.

        Returns:
            A list of list of dictionaries of results, where the outer list
            corresponds to separate calls to `async_update`, and the inner
            list corresponds to the results from each Learner(s). Or if the results
            are reduced, a list of dictionaries of the reduced results from each
            call to async_update that is ready.
        """
    if self.is_local:
        raise ValueError('Cannot call `async_update` when running in local mode with num_workers=0.')
    else:
        if minibatch_size is not None:
            minibatch_size //= len(self._workers)

        def _learner_update(learner, minibatch):
            return learner.update(minibatch, minibatch_size=minibatch_size, num_iters=num_iters, reduce_fn=reduce_fn)
        if len(self._in_queue) == self._in_queue.maxlen:
            self._in_queue_ts_dropped += len(self._in_queue[0])
        self._in_queue.append(batch)
        results = self._worker_manager.fetch_ready_async_reqs(tags=list(self._inflight_request_tags))
        if self._worker_manager_ready():
            count = 0
            while len(self._in_queue) > 0 and count < 3:
                update_tag = str(uuid.uuid4())
                self._inflight_request_tags.add(update_tag)
                batch = self._in_queue.popleft()
                self._worker_manager.foreach_actor_async([partial(_learner_update, minibatch=minibatch) for minibatch in ShardBatchIterator(batch, len(self._workers))], tag=update_tag)
                count += 1
        results = self._get_async_results(results)
        if reduce_fn is None:
            return results
        else:
            return [reduce_fn(r) for r in results]