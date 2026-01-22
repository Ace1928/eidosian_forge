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
def setup_retriever(opt):
    print('[ Setting up Retriever ]')
    task = 'wikipedia:full'
    ret_opt = copy.deepcopy(opt)
    ret_opt['model_file'] = 'models:wikipedia_full/tfidf_retriever/model'
    ret_opt['retriever_num_retrieved'] = opt.get('num_passages_retrieved', 7)
    ret_opt['retriever_mode'] = 'keys'
    ret_opt['override'] = {'remove_title': False}
    ir_agent = create_agent(ret_opt)
    return (ir_agent, task)