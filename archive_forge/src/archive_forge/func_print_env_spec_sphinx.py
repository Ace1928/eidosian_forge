import json
from typing import Union
import gym
from minerl.herobraine.env_spec import EnvSpec
def print_env_spec_sphinx(env_spec: EnvSpec) -> None:
    env = env_spec()
    env_name = env.name
    print('_' * len(env_name))
    print(f'{env_name}')
    print('_' * len(env_name))
    print(env.__doc__)
    if hasattr(env, 'inventory'):
        print('..................')
        print('Starting Inventory')
        print('..................')
        starting_inv_canonical = {}
        for stack in env.inventory:
            item_id = stack['type']
            starting_inv_canonical[item_id] = stack['quantity']
        print(_format_dict(starting_inv_canonical))
    if hasattr(env, 'max_episode_steps'):
        print('..................')
        print('Max Episode Steps')
        print('..................')
        print(f':code:`{env.max_episode_steps}`')
    print('\n.....')
    print('Usage')
    print('.....')
    usage_str = f'.. code-block:: python\n    \n        env = gym.make("{env.name}")\n    '
    print(usage_str)