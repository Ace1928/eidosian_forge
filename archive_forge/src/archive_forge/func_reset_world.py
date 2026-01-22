from parlai.core.worlds import BatchWorld, DynamicBatchWorld
from parlai.utils.misc import msg_to_str
from parlai.utils.conversations import Conversations
import parlai.utils.logging as logging
import copy
from tqdm import tqdm
def reset_world(self, idx=0):
    self._add_episode(self._current_episodes[idx])
    self._current_episodes[idx] = []