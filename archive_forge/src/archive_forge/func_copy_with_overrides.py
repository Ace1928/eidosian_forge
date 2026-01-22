import copy
from typing import Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import EnvConfigDict
def copy_with_overrides(self, env_config: Optional[EnvConfigDict]=None, worker_index: Optional[int]=None, vector_index: Optional[int]=None, remote: Optional[bool]=None, num_workers: Optional[int]=None, recreated_worker: Optional[bool]=None) -> 'EnvContext':
    """Returns a copy of this EnvContext with some attributes overridden.

        Args:
            env_config: Optional env config to use. None for not overriding
                the one from the source (self).
            worker_index: Optional worker index to use. None for not
                overriding the one from the source (self).
            vector_index: Optional vector index to use. None for not
                overriding the one from the source (self).
            remote: Optional remote setting to use. None for not overriding
                the one from the source (self).
            num_workers: Optional num_workers to use. None for not overriding
                the one from the source (self).
            recreated_worker: Optional flag, indicating, whether the worker that holds
                the env is a recreated one. This means that it replaced a previous
                (failed) worker when `recreate_failed_workers=True` in the Algorithm's
                config.

        Returns:
            A new EnvContext object as a copy of self plus the provided
            overrides.
        """
    return EnvContext(copy.deepcopy(env_config) if env_config is not None else self, worker_index if worker_index is not None else self.worker_index, vector_index if vector_index is not None else self.vector_index, remote if remote is not None else self.remote, num_workers if num_workers is not None else self.num_workers, recreated_worker if recreated_worker is not None else self.recreated_worker)