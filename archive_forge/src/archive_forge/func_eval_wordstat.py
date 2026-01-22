from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.metrics import normalize_answer
from parlai.core.logs import TensorboardLogger
from controllable_seq2seq.controls import (
from controllable_seq2seq.util import ConvAI2History
from collections import Counter
import copy
import random
import json
import time
import os
def eval_wordstat(opt):
    """
    Evaluates a model.

    :param opt: tells the evaluation function how to run
    """
    random.seed(42)
    initialize_control_information(opt)
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)
    if opt.get('external_dict'):
        print('[ Using external dictionary from: {} ]'.format(opt['external_dict']))
        dict_opt = copy.deepcopy(opt)
        dict_opt['dict_file'] = opt['external_dict']
        dictionary = DictionaryAgent(dict_opt)
    else:
        print('[ Using model bundled dictionary ]')
        dictionary = agent.dict
    batch_size = opt['batchsize']
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()
    data = {}
    data['opt'] = agent.opt
    if opt['gold_response']:
        model_dir, _ = os.path.split(opt.get('model_file'))
        outfile = os.path.join(model_dir, 'goldresponse')
        if opt['use_reply'] != 'label':
            raise ValueError('You should set --use-reply label (not --use-reply model) when measuring goldresponse stats')
    else:
        outfile = '%s.%s.%s.%s' % (opt.get('model_file'), opt.get('datatype'), 'use%sreply' % agent.opt['use_reply'], 'beam%i' % agent.opt['beam_size'])
        if agent.opt['beam_size'] > 1:
            outfile += '.beamminnbest%i' % agent.opt['beam_min_n_best']
        if len(agent.control_settings) > 0:
            outfile += '.setcontrols:' + '_'.join(['%s%s' % (c, str(agent.control_settings[c]['set_value'])) for c in sorted(agent.control_settings.keys())])
        if agent.opt['beam_reorder'] not in ['none', False]:
            outfile += '.beamreorder_%s' % agent.opt['beam_reorder']
        if len(agent.wd_features) > 0:
            sorted_bfw = sorted(list(zip(agent.wd_features, agent.wd_wts)), key=lambda x: x[0])
            outfile += '.WDfeatures:' + '_'.join(['%s%s' % (f, str(w)) for f, w in sorted_bfw])
    if opt['num_examples'] != -1:
        outfile += '.numex%i' % opt['num_examples']
    outfile += '.wordstats.json'
    print('\nOutfile: %s\n' % outfile)
    cnt = 0
    word_statistics = {'mean_wlength': [], 'mean_clength': [], 'freqs_cnt': Counter(), 'word_cnt': 0, 'pred_list': [], 'pure_pred_list': [], 'context_list': []}
    bins = [int(i) for i in opt['freq_bins'].split(',')]
    sent_attrs = {attr: [] for attr in ATTR2SENTSCOREFN.keys()}
    histories = []

    def process_prediction(prediction, word_statistics):
        word_statistics['pred_list'].append(normalize_answer(prediction))
        freqs, _cnt, wlength, clength = get_word_stats(prediction, dictionary, bins=bins)
        word_statistics['word_cnt'] += _cnt
        word_statistics['mean_wlength'].append(wlength)
        word_statistics['mean_clength'].append(clength)
        word_statistics['freqs_cnt'] += Counter(freqs)
        return word_statistics
    t0 = time.time()
    while not world.epoch_done():
        world.parley()
        assert batch_size != 1
        for w in world.worlds:
            try:
                try:
                    response_act = w.acts[-1]
                    prediction = response_act['text']
                except KeyError:
                    continue
                if opt['gold_response']:
                    prediction = w.acts[0]['eval_labels'][0]
                    response_act = {'text': prediction}
                word_statistics['context_list'].append(w.acts[0]['text'])
                word_statistics['pure_pred_list'].append(prediction)
            except IndexError:
                continue
            cnt += 1
            word_statistics = process_prediction(prediction, word_statistics)
            history = ConvAI2History(w.acts[0]['text'])
            histories.append(history)
            sent_attrs = update_sent_attr_stats(sent_attrs, history, prediction)
        if log_time.time() > log_every_n_secs:
            report = world.report()
            text, report = log_time.log(report['exs'], world.num_examples(), report)
            print(text)
        if opt['num_examples'] > 0 and cnt >= opt['num_examples']:
            break
    if world.epoch_done():
        print('EPOCH DONE')
    print('Time to process %i examples: %f seconds' % (cnt, time.time() - t0))
    unique_list = []
    cntr = Counter(word_statistics['pred_list'])
    for k, v in cntr.items():
        if v == 1:
            unique_list.append(k)
    unique_percent = len(unique_list) / len(word_statistics['pred_list']) * 100
    report = world.report()
    if opt['gold_response']:
        report['ppl'] = 0.0
    print(report)
    data['unique_percent'] = unique_percent
    data['word_statistics'] = word_statistics
    data['report'] = report
    data['histories'] = [(hist.persona_lines, hist.partner_utts, hist.own_utts) for hist in histories]
    data['sent_attrs'] = sent_attrs
    print('Writing to %s...' % outfile)
    with open(outfile, 'w') as f:
        json.dump(data, f)