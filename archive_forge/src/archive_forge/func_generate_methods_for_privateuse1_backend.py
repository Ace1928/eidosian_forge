import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
def generate_methods_for_privateuse1_backend(for_tensor: bool=True, for_module: bool=True, for_storage: bool=False, unsupported_dtype: Optional[List[torch.dtype]]=None) -> None:
    """
    Automatically generate attributes and methods for the custom backend after rename privateuse1 backend.

    In the default scenario, storage-related methods will not be generated automatically.

    When you implement kernels for various torch operations, and register them to the PrivateUse1 dispatch key.
    And call the function torch.rename_privateuse1_backend("foo") to rename your backend name.
    At this point, you can easily register specific methods and attributes by calling this function.
    Just like torch.Tensor.foo(), torch.Tensor.is_foo, torch.Storage.foo(), torch.Storage.is_foo.

    Note: We recommend you use generic functions (check devices are equal or to(device=)).
    We provide these methods for convenience only and they will be "monkey patched" onto the objects
    and so will not be properly typed. For Storage methods generate, if you need to support sparse data storage,
    you need to extend the implementation yourself.

    Args:
        for_tensor (bool): whether register related methods for torch.Tensor class.
        for_module (bool): whether register related methods for torch.nn.Module class.
        for_storage (bool): whether register related methods for torch.Storage class.
        unsupported_dtype (List[torch.dtype]): takes effect only when the storage method needs to be generated,
            indicating that the storage does not support the torch.dtype type.

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.rename_privateuse1_backend("foo")
        >>> torch.utils.generate_methods_for_privateuse1_backend()
        # Then automatically generate backend-related attributes and methods.
        >>> a = torch.tensor(2).foo()
        >>> a.is_foo
        >>> hasattr(torch.nn.Module, 'foo')
    """
    custom_backend_name = _get_privateuse1_backend_name()
    if for_tensor:
        _generate_tensor_methods_for_privateuse1_backend(custom_backend_name)
    if for_module:
        _generate_module_methods_for_privateuse1_backend(custom_backend_name)
    if for_storage:
        _generate_storage_methods_for_privateuse1_backend(custom_backend_name, unsupported_dtype)