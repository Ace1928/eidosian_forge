from typing import Optional
import warnings
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from .planner import SavePlanner
from .default_planner import DefaultSavePlanner
from .storage import (
from .metadata import Metadata, STATE_DICT_TYPE
from .utils import _DistWrapper

    Save a distributed model in SPMD style.

    This function is different from ``torch.save()`` as it handles
    ``ShardedTensor`` , and ``DTensor`` by having each rank only save their local shards.

    For each ``Stateful`` object (having both a ``state_dict`` and a ``load_state_dict``),
    save will call ``state_dict`` before serialization.

    .. warning::
        There is no guarantees of Backwards Compatibility across PyTorch versions
        for saved state_dicts.

    .. warning::
        If using the `process_group` argument, make sure that only its ranks
        call `save_state_dict` and that all data in state_dict belong to it.

    .. note::
        When saving checkpoint for FSDP's `ShardingStrategy.HYBRID_SHARD`, only one of
        the shard_group should be calling `save_state_dict` and the corresponding process
        group needs to be passed in.

    .. note::
        This function can be used to save a state_dict without having a process group
        initialized by passing ``no_dist=True``.


    Args:
        state_dict (Dict[str, Any]): The state_dict to save.
        storage_writer (StorageWriter):
            Instance of StorageWrite use to perform writes.
        process_group (ProcessGroup):
            ProcessGroup to be used for cross-rank synchronization.
        coordinator_rank (int): Rank to use to coordinate the checkpoint.
            rank0 is used by default.
        no_dist (bool): If ``True``, distributed checkpoint will not save
            in SPMD style. (Default: ``False``)

    Returns:
        Metadata: Metadata object for the saved checkpoint.

    Example:
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()

        >>> model_state_dict = my_model.state_dict()

        >>> fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter("/checkpoint/1")
        >>> torch.distributed.checkpoint.save_state_dict(
        >>>     state_dict=model_state_dict,
        >>>     storage_writer=fs_storage_writer,
        >>> )

    .. note::
        save_state_dict uses collectives to coordinate writes across ranks.
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication takes place.
        In this case, the device used is given by ``torch.cuda.current_device()``
        and it is the user's responsibility to ensure that this is set so that
        each rank has an individual GPU, via ``torch.cuda.set_device()``.
    