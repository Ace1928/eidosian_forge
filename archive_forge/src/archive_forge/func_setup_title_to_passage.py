from parlai.core.agents import create_agent
from parlai.core.teachers import create_task_agent_from_taskname
from parlai.core.params import ParlaiParser
from parlai.utils.misc import AttrDict
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import (
from task_config import task_config
from parlai.core.dict import DictionaryAgent
import os
import copy
import tqdm
import pickle
import parlai.core.build_data as build_data
from urllib.parse import unquote
def setup_title_to_passage(opt):
    print('[ Setting up Title to Passage Dict ]')
    saved_dp = os.path.join(os.getcwd() + '/data/', 'title_to_passage.pkl')
    if os.path.exists(saved_dp):
        print('[ Loading from saved location, {} ]'.format(saved_dp))
        with open(saved_dp, 'rb') as f:
            title_to_passage = pickle.load(f)
            return title_to_passage
    topics_path = '{}/personas_with_wiki_links.txt'.format(os.getcwd())
    topics = []
    with open(topics_path) as f:
        text = f.read()
        personas = text.split('\n\n')
        for persona in personas:
            persona = persona.split('\n')
            for i in range(1, len(persona)):
                p_i = persona[i]
                if 'https' in p_i:
                    topic = unquote(p_i[p_i.rfind('/') + 1:]).replace('_', ' ')
                    topics.append(topic)
    ordered_opt = opt.copy()
    ordered_opt['datatype'] = 'train:ordered:stream'
    ordered_opt['batchsize'] = 1
    ordered_opt['task'] = 'wikipedia:full:key-value'
    teacher = create_task_agent_from_taskname(ordered_opt)[0]
    title_to_passage = {}
    i = 0
    length = teacher.num_episodes()
    pbar = tqdm.tqdm(total=length)
    while not teacher.epoch_done():
        pbar.update(1)
        i += 1
        action = teacher.act()
        title = action['text']
        if title in topics:
            text = action['labels'][0]
            title_to_passage[title] = text
    pbar.close()
    print('[ Finished Building Title to Passage dict; saving now]')
    with open(saved_dp, 'wb') as f:
        pickle.dump(title_to_passage, f)
    return title_to_passage