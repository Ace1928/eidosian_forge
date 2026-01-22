from typing import List
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import TensorType, TensorStructType
def unbatch_repeat_dim(self) -> List[TensorType]:
    """Unbatches the repeat dimension (the one `max_len` in size).

        This removes the repeat dimension. The result will be a Python list of
        with length `self.max_len`. Note that the data is still padded.

        .. testcode::
            :skipif: True

            batch = RepeatedValues(<Tensor shape=(B, N, K)>)
            items = batch.unbatch()
            len(items) == batch.max_len

        .. testoutput::

            True

        .. testcode::
            :skipif: True

            print(items)

        .. testoutput::

            [<Tensor_1 shape=(B, K)>, ..., <Tensor_N shape=(B, K)>]
        """
    return _unbatch_helper(self.values, self.max_len)