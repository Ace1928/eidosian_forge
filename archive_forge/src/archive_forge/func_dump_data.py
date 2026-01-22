from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.conversations import Conversations
from parlai.utils.misc import TimeLogger
import random
import tempfile
def dump_data(opt):
    """
    Dump task data to ACUTE-Eval.
    """
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    task = opt.get('task')
    speaker_0_id = opt.get('speaker_0_id') or f'{task}_as_human'
    speaker_1_id = opt.get('speaker_1_id') or f'{task}_as_model'
    if opt['outfile'] is None:
        outfile = tempfile.mkstemp(prefix='{}_{}_'.format(opt['task'], opt['datatype']), suffix='.txt')[1]
    else:
        outfile = opt['outfile']
    num_episodes = world.num_episodes() if opt['num_episodes'] == -1 else min(opt['num_episodes'], world.num_episodes())
    log_timer = TimeLogger()
    print(f'[ starting to convert, saving output to {outfile} ]')
    dialogues = []
    for _ in range(num_episodes):
        episode = []
        episode_done = False
        while not episode_done:
            world.parley()
            acts = world.get_acts()
            text = acts[0].get('text')
            split_text = text.split('\n')
            label = random.choice(acts[0].get('labels', acts[0].pop('eval_labels', None)))
            if not episode and opt.get('prepended_context'):
                context = split_text[:-1]
                text = split_text[-1]
                context_turn = [{'text': context, 'episode_done': False, 'id': 'context'} for _ in range(2)]
                episode.append(context_turn)
            turn = [{'text': text, 'episode_done': False, 'id': speaker_0_id}, {'text': label, 'episode_done': False, 'id': speaker_1_id}]
            episode.append(turn)
            if acts[0].get('episode_done', False):
                episode[-1][-1]['episode_done'] = True
                episode_done = True
                dialogues.append(episode)
            if log_timer.time() > opt['log_every_n_secs']:
                text, _log = log_timer.log(world.total_parleys, world.num_examples())
                print(text)
        if world.epoch_done():
            break
    Conversations.save_conversations(dialogues, outfile, opt)