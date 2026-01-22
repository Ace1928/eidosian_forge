from typing import List, Optional, Sequence
import gym
from minerl.env import _fake, _singleagent
from minerl.herobraine import wrappers
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs import simple_embodiment
from minerl.herobraine.hero import handlers, mc
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
def get_blacklist_reason(self, npz_data: dict) -> Optional[str]:
    """
        Some saved demonstrations are bogus -- they only contain lobby frames.

        We can automatically skip these by checking for whether any snowballs
        were thrown.
        """
    equip = npz_data.get('observation$equipped_items$mainhand$type')
    use = npz_data.get('action$use')
    if equip is None:
        return f'Missing equip observation. Available keys: {list(npz_data.keys())}'
    if use is None:
        return f'Missing use action. Available keys: {list(npz_data.keys())}'
    assert len(equip) == len(use) + 1, (len(equip), len(use))
    for i in range(len(use)):
        if use[i] == 1 and equip[i] == 'snowball':
            return None
    return 'BasaltEnv never threw a snowball'