import json
import os
import random
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from parlai.projects.self_feeding.utils import Parley
def create_supp(opt):
    """
    Evaluates a model.

    :param opt: tells the evaluation function how to run
    :return: the final result of calling report()
    """
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)
    num_seen = 0
    num_misses = 0
    num_supp = 0
    num_supp_correct = 0
    examples = []
    while not world.epoch_done():
        world.parley()
        num_seen += 1
        if num_seen % 1000 == 0:
            print(f'{num_seen}/{world.num_examples()}')
        report = world.report()
        if report['accuracy'] < 1.0:
            num_misses += 1
            if random.random() < opt['conversion_rate']:
                num_supp += 1
                texts = world.acts[0]['text'].split('\n')
                context = texts[-1]
                memories = texts[:-1]
                candidates = world.acts[0]['label_candidates']
                reward = 1
                if random.random() < opt['conversion_acc']:
                    num_supp_correct += 1
                    response = world.acts[0]['eval_labels'][0]
                else:
                    response = random.choice(world.acts[0]['label_candidates'][:NUM_INLINE_CANDS - 1])
                example = Parley(context, response, reward, candidates, memories)
                examples.append(example)
        world.reset_metrics()
    print('EPOCH DONE')
    print(f'Model file: {opt['model_file']}')
    print(f'Deploy file: {opt['task']}')
    print(f'Supp file: {opt['outfile']}')
    print(f'Deploy size (# examples seen): {num_seen}')
    print(f'Supp size (# examples converted): {num_supp}')
    acc = 1 - num_misses / num_seen
    print(f'Accuracy (% of deploy): {acc * 100:.1f}% ({num_misses} misses)')
    print(f'Conversion rate (% of misses): {num_supp / num_misses * 100:.2f}% ({num_supp}/{num_misses})')
    print(f'Conversion acc (% of converted): {num_supp_correct / num_supp * 100:.2f}% ({num_supp_correct}/{num_supp})')
    with open(opt['outfile'], 'w') as outfile:
        for ex in examples:
            outfile.write(json.dumps(ex.to_dict()) + '\n')