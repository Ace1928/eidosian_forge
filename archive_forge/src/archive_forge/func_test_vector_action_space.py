import collections
from functools import reduce
from minerl.herobraine.hero.spaces import Dict
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
import minerl.herobraine.wrappers as wrappers
import minerl.herobraine.envs as envs
from minerl.herobraine.wrappers.util import union_spaces
import numpy as np
def test_vector_action_space(base_env=envs.MINERL_OBTAIN_DIAMOND_V0, common_env=None):
    vec_env = wrappers.Vectorized(base_env, common_env)
    assert isinstance(vec_env.action_space, Dict)
    assert isinstance(vec_env.observation_space, Dict)
    print(vec_env.action_space)
    print(vec_env.action_space.spaces)
    assert 'vector' in vec_env.action_space.spaces
    assert 'vector' in vec_env.observation_space.spaces