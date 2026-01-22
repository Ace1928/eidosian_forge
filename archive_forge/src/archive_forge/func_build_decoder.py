import gymnasium as gym
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.base import Encoder, Model
from ray.rllib.utils import override
def build_decoder(self, framework: str) -> Model:
    """Builds the World-Model's decoder network depending on the obs space."""
    if framework != 'tf2':
        raise NotImplementedError
    if self.is_img_space:
        from ray.rllib.algorithms.dreamerv3.tf.models.components import conv_transpose_atari
        return conv_transpose_atari.ConvTransposeAtari(model_size=self.model_size, gray_scaled=self.is_gray_scale)
    else:
        from ray.rllib.algorithms.dreamerv3.tf.models.components import vector_decoder
        return vector_decoder.VectorDecoder(model_size=self.model_size, observation_space=self.observation_space)