from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from controllable_seq2seq.controls import sort_into_bucket
from collections import Counter
import random
def bucket_data(opt):
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    if opt['num_examples'] == -1:
        num_examples = world.num_examples()
    else:
        num_examples = opt['num_examples']
    log_timer = TimeLogger()
    assert opt['control'] != ''
    ctrl = opt['control']
    num_buckets = opt['num_buckets']
    ctrl_vals = []
    for _ in range(num_examples):
        world.parley()
        world.acts[0]['labels'] = world.acts[0].get('labels', world.acts[0].pop('eval_labels', None))
        if ctrl not in world.acts[0].keys():
            raise Exception("Error: control %s isn't in the data. available keys: %s" % (ctrl, ', '.join(world.acts[0].keys())))
        ctrl_val = world.acts[0][ctrl]
        if ctrl_val == 'None':
            assert ctrl == 'lastuttsim'
            ctrl_val = None
        else:
            ctrl_val = float(ctrl_val)
        if ctrl == 'avg_nidf':
            assert ctrl_val >= 0
            assert ctrl_val <= 1
        elif ctrl == 'question':
            assert ctrl_val in [0, 1]
        elif ctrl == 'lastuttsim':
            if ctrl_val is not None:
                assert ctrl_val >= -1
                assert ctrl_val <= 1
        else:
            raise Exception('Unexpected ctrl name: %s' % ctrl)
        ctrl_vals.append(ctrl_val)
        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(world.total_parleys, world.num_examples())
            print(text)
        if world.epoch_done():
            print('EPOCH DONE')
            break
    if ctrl == 'lastuttsim':
        num_nones = len([v for v in ctrl_vals if v is None])
        ctrl_vals = [v for v in ctrl_vals if v is not None]
        print('Have %i Nones for lastuttsim; these have been removed for bucket calculation' % num_nones)
    print('Collected %i control vals between %.6f and %.6f' % (len(ctrl_vals), min(ctrl_vals), max(ctrl_vals)))
    print('Calculating lowerbounds for %i buckets...' % num_buckets)
    ctrl_vals = sorted(ctrl_vals)
    lb_indices = [int(len(ctrl_vals) * i / num_buckets) for i in range(num_buckets)]
    lbs = [ctrl_vals[idx] for idx in lb_indices]
    print('\nBucket lowerbounds for control %s: ' % ctrl)
    print(lbs)
    bucket_sizes = Counter()
    bucket_ids = [sort_into_bucket(ctrl_val, lbs) for ctrl_val in ctrl_vals]
    bucket_sizes.update(bucket_ids)
    print('\nBucket sizes: ')
    for bucket_id in sorted(bucket_sizes.keys()):
        print('%i: %i' % (bucket_id, bucket_sizes[bucket_id]))