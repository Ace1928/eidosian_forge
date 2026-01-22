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
def test_env(environment='MineRLObtainTest-v0', interactive=False):
    if not interactive:
        pass
    inst = InstanceManager.add_existing_instance(9001)
    env = gym.make(environment, instances=[inst])
    done = False
    inventories = []
    rewards = []
    for _ in range(1):
        env.reset()
        total_reward = 0
        print_next_inv = False
        action = env.action_space.no_op()
        action['equip'] = 'red_flower'
        obs, _, _, _ = env.step(action)
        obs, _, _, _ = env.step(env.action_space.no_op())
        assert obs['equipped_items']['mainhand']['type'] == 'other', '{} is not of type other'.format(obs['equipped_items']['mainhand']['type'])
        for action in gen_obtain_debug_actions(env):
            for key, value in action.items():
                if isinstance(value, str) and value in reward_dict and (key not in ['equip']):
                    print('Action of {}:{} if successful gets {}'.format(key, value, reward_dict[value]))
            obs, reward, done, info = env.step(action)
            env.render()
            if print_next_inv:
                print(obs['inventory'])
                print_next_inv = False
            if interactive:
                key = input('')
            if reward != 0:
                print(obs['inventory'])
                print(reward)
                print_next_inv = True
                total_reward += reward
            if done:
                break
        print('MISSION DONE')
        inventories.append(obs['inventory'])
        rewards.append(total_reward)
    for r, i in zip(inventories, rewards):
        print(r)
        print(i)
    if environment == 'MineRLObtainTest-v0':
        assert all((r == 1482.0 for r in rewards))
    elif environment == 'MineRLObtainTestDense-v0':
        assert all((r == 2874.0 for r in rewards))