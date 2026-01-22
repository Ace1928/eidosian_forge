from typing import List, Optional, Union
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.types import kernel_signature
from torchgen.context import with_native_function_and_index
from torchgen.model import BackendIndex, NativeFunction, NativeFunctionsGroup
from torchgen.utils import mapMaybe
@with_native_function_and_index
def gen_unstructured(f: NativeFunction, backend_index: BackendIndex) -> Optional[str]:
    sig = kernel_signature(f, backend_index)
    metadata = backend_index.get_kernel(f)
    if metadata is None:
        return None
    if 'legacy::' in metadata.kernel:
        return None
    else:
        prefix = 'static' if backend_index.external else 'TORCH_API'
        return f'{prefix} {sig.decl(name=metadata.kernel)};'