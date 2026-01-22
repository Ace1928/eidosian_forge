import jax
from keras.src.backend.config import floatx
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed
def jax_draw_seed(seed):
    if isinstance(seed, jax.Array):
        return seed
    else:
        return draw_seed(seed)