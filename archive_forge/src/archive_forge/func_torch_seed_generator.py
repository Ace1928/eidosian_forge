import torch
import torch._dynamo as dynamo
import torch.nn.functional as tnn
from keras.src.backend.config import floatx
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.core import to_torch_dtype
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed
@dynamo.disable()
def torch_seed_generator(seed):
    first_seed, second_seed = draw_seed(seed)
    device = get_device()
    if device == 'meta':
        return None
    generator = torch.Generator(device=get_device())
    generator.manual_seed(int(first_seed + second_seed))
    return generator