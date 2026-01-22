import pickle
import ctypes
import os
from ..ndarray import NDArray
from ..ndarray import _ndarray_cls
from ..base import _LIB, c_str
from ..base import check_call, mx_uint, py_str
from ..base import NDArrayHandle, KVStoreHandle
from .. import optimizer as opt
from .base import _ctype_key_value, _ctype_dict, KVStoreBase
def set_gradient_compression(self, compression_params):
    """ Specifies type of low-bit quantization for gradient compression          and additional arguments depending on the type of compression being used.

        2bit Gradient Compression takes a positive float `threshold`.
        The technique works by thresholding values such that positive values in the
        gradient above threshold will be set to threshold. Negative values whose absolute
        values are higher than threshold, will be set to the negative of threshold.
        Values whose absolute values are less than threshold will be set to 0.
        By doing so, each value in the gradient is in one of three states. 2bits are
        used to represent these states, and every 16 float values in the original
        gradient can be represented using one float. This compressed representation
        can reduce communication costs. The difference between these thresholded values and
        original values is stored at the sender's end as residual and added to the
        gradient in the next iteration.

        When kvstore is 'local', gradient compression is used to reduce communication
        between multiple devices (gpus). Gradient is quantized on each GPU which
        computed the gradients, then sent to the GPU which merges the gradients. This
        receiving GPU dequantizes the gradients and merges them. Note that this
        increases memory usage on each GPU because of the residual array stored.

        When kvstore is 'dist', gradient compression is used to reduce communication
        from worker to sender. Gradient is quantized on each worker which
        computed the gradients, then sent to the server which dequantizes
        this data and merges the gradients from each worker. Note that this
        increases CPU memory usage on each worker because of the residual array stored.
        Only worker to server communication is compressed in this setting.
        If each machine has multiple GPUs, currently this GPU to GPU or GPU to CPU communication
        is not compressed. Server to worker communication (in the case of pull)
        is also not compressed.

        To use 2bit compression, we need to specify `type` as `2bit`.
        Only specifying `type` would use default value for the threshold.
        To completely specify the arguments for 2bit compression, we would need to pass
        a dictionary which includes `threshold` like:
        {'type': '2bit', 'threshold': 0.5}

        Parameters
        ----------
        compression_params : dict
            A dictionary specifying the type and parameters for gradient compression.
            The key `type` in this dictionary is a
            required string argument and specifies the type of gradient compression.
            Currently `type` can be only `2bit`
            Other keys in this dictionary are optional and specific to the type
            of gradient compression.
        """
    if 'device' in self.type or 'dist' in self.type:
        ckeys, cvals = _ctype_dict(compression_params)
        check_call(_LIB.MXKVStoreSetGradientCompression(self.handle, mx_uint(len(compression_params)), ckeys, cvals))
    else:
        raise Exception('Gradient compression is not supported for this type of kvstore')