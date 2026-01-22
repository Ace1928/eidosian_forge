import numpy as np
from ..parser import convert_to_valid_einsum_chars
from ..sharing import to_backend_cache_wrap
def torch_contract(*arrays):
    torch_arrays = [to_torch(x) for x in arrays]
    torch_out = expr._contract(torch_arrays, backend='torch')
    if torch_out.device.type == 'cpu':
        return torch_out.numpy()
    return torch_out.cpu().numpy()