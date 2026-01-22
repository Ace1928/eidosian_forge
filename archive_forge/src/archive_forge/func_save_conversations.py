import datetime
import json
import os
import random
from parlai.utils.misc import AttrDict
import parlai.utils.logging as logging
@classmethod
def save_conversations(cls, act_list, datapath, opt, save_keys='all', context_ids='context', self_chat=False, **kwargs):
    """
        Write Conversations to file from an act list.

        Conversations assume the act list is of the following form: a list of episodes,
        each of which is comprised of a list of act pairs (i.e. a list dictionaries
        returned from one parley)
        """
    to_save = cls._get_path(datapath)
    context_ids = context_ids.split(',')
    speakers = []
    with open(to_save, 'w') as f:
        for ep in act_list:
            if not ep:
                continue
            convo = {'dialog': [], 'context': [], 'metadata_path': Metadata._get_path(to_save)}
            for act_pair in ep:
                new_pair = []
                for ex in act_pair:
                    ex_id = ex.get('id')
                    if ex_id in context_ids:
                        context = True
                    else:
                        context = False
                        if ex_id not in speakers:
                            speakers.append(ex_id)
                    turn = {}
                    if save_keys != 'all':
                        save_keys_lst = save_keys.split(',')
                    else:
                        save_keys_lst = [key for key in ex.keys() if key != 'metrics']
                    for key in save_keys_lst:
                        turn[key] = ex.get(key, '')
                    turn['id'] = ex_id
                    if not context:
                        new_pair.append(turn)
                    else:
                        convo['context'].append(turn)
                if new_pair:
                    convo['dialog'].append(new_pair)
            json_convo = json.dumps(convo)
            f.write(json_convo + '\n')
    logging.info(f'Conversations saved to file: {to_save}')
    Metadata.save_metadata(to_save, opt, self_chat=self_chat, speakers=speakers, **kwargs)