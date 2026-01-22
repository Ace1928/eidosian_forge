from parlai.core.worlds import BatchWorld, DynamicBatchWorld
from parlai.utils.misc import msg_to_str
from parlai.utils.conversations import Conversations
import parlai.utils.logging as logging
import copy
from tqdm import tqdm
def write_conversations_format(self, outfile, world):
    logging.info(f'Saving log to {outfile} in Conversations format')
    Conversations.save_conversations(self._logs, outfile, world.opt, self_chat=world.opt.get('selfchat_task', False))