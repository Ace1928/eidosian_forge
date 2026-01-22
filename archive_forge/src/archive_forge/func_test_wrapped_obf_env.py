from minerl.env.malmo import InstanceManager
import minerl
import time
import gym
import numpy as np
import logging
import coloredlogs
from minerl.herobraine.wrappers.vector_wrapper import Vectorized
from minerl.herobraine.env_specs.obtain_specs import ObtainDiamondDebug
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
from minerl.herobraine.wrappers.obfuscation_wrapper import Obfuscated
import minerl.herobraine.envs as envs
import minerl.herobraine
def test_wrapped_obf_env():
    return test_wrapped_env(environment='MineRLObtainTest-v0', wrapped_env='MineRLObtainTestVectorObf-v0')