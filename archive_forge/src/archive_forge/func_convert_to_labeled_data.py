from parlai.core.worlds import BatchWorld, DynamicBatchWorld
from parlai.utils.misc import msg_to_str
from parlai.utils.conversations import Conversations
import parlai.utils.logging as logging
import copy
from tqdm import tqdm
def convert_to_labeled_data(self, episode):
    out = []
    text_lst = []
    for parley in episode:
        first_act, second_act = parley
        if 'text' in first_act:
            text_lst.append(first_act['text'])
        if second_act.get('id') != 'context':
            label = second_act.get('text')
            out.append({'id': first_act.get('id', ''), 'text': '\n'.join(text_lst), 'labels': [label], 'episode_done': False})
            text_lst = []
    if len(out) > 0:
        out[-1]['episode_done'] = True
    return out