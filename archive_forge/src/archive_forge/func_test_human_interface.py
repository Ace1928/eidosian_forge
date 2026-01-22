import logging
import minerl
import gym
from minerl.human_play_interface.human_play_interface import HumanPlayInterface
import coloredlogs
def test_human_interface():
    env = gym.make(ENV_NAMES[3])
    env = HumanPlayInterface(env)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step()
    print('Episode done')
    env.close()