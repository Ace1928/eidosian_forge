from typing import Callable, Dict, Type
import importlib
from collections import namedtuple
def load_agent_module(agent_path: str):
    """
    Return the module for an agent specified by ``--model``.

    Can be formatted in several different ways:

    * full: `-m parlai.agents.seq2seq.seq2seq:Seq2seqAgent`
    * shorthand: -m seq2seq, which will check both paths
      ``parlai.agents.seq2seq.seq2seq:Seq2seqAgent`` and
      ``parlai.agents.seq2seq.agents:Seq2seqAgent``
    * half-shorthand: ``-m seq2seq/variant``, which will check the path
      `parlai.agents.seq2seq.variant:VariantAgent`

    The base path to search when using shorthand formats can be changed from
    "parlai" to "parlai_internal" by prepending "internal:" to the path, e.g.
    "internal:seq2seq".

    To use agents in projects, you can prepend "projects:" and the name of the
    project folder to model arguments, e.g. "projects:personachat:kvmemnn"
    will translate to ``projects/personachat/kvmemnn``.

    :param agent_path:
        path to model class in one of the above formats.

    :return:
        module of agent
    """
    global AGENT_REGISTRY
    if agent_path in AGENT_REGISTRY:
        return AGENT_REGISTRY[agent_path]
    repo = 'parlai'
    if agent_path.startswith('internal:'):
        repo = 'parlai_internal'
        agent_path = agent_path[9:]
    elif agent_path.startswith('fb:'):
        repo = 'parlai_fb'
        agent_path = agent_path[3:]
    if agent_path.startswith('projects:'):
        path_list = agent_path.split(':')
        if len(path_list) != 3:
            raise RuntimeError('projects paths should follow pattern projects:folder:model; you used {}'.format(agent_path))
        folder_name = path_list[1]
        model_name = path_list[2]
        module_name = 'projects.{p}.{m}.{m}'.format(m=model_name, p=folder_name)
        class_name = _name_to_agent_class(model_name)
    elif ':' in agent_path:
        path_list = agent_path.split(':')
        module_name = path_list[0]
        class_name = path_list[1]
    elif '/' in agent_path:
        path_list = agent_path.split('/')
        module_name = '%s.agents.%s.%s' % (repo, path_list[0], path_list[1])
        class_name = _name_to_agent_class(path_list[1])
    else:
        class_name = _name_to_agent_class(agent_path)
        try:
            module_name = '%s.agents.%s.agents' % (repo, agent_path)
            importlib.import_module(module_name)
        except ImportError:
            module_name = '%s.agents.%s.%s' % (repo, agent_path, agent_path)
    my_module = importlib.import_module(module_name)
    model_class = getattr(my_module, class_name)
    return model_class