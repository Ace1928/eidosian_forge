from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.utils import seeding
import ray
def get_params(self, rng):
    return {'MASSCART': rng.uniform(low=0.5, high=2.0)}