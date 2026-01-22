import random
import io
import os
import pickle
from parlai.utils.misc import msg_to_str
def write_dialog(opt, fw, d, label_type, split):
    l = len(d['speech'])
    msgs = []
    text = ''
    d['which'] = []
    for i in range(0, len(d['emote'])):
        lab = 'none'
        if d['emote'][i] is not None:
            lab = 'emote'
        if d['action'][i] is not None:
            lab = 'action'
        d['which'].append(lab)
    if opt.get('light_use_taskname', True):
        text = '_task_' + label_type + '\n'
    if opt['light_use_setting']:
        text += '_setting_name ' + d['setting']['name'] + ', ' + d['setting']['category'] + '\n'
        text += '_setting_desc ' + d['setting']['description'] + '\n'
    if opt['light_use_person_names']:
        text += '_partner_name ' + d['partner_agent']['name'] + '\n'
        text += '_self_name ' + d['self_agent']['name'] + '\n'
    if use_feat(opt, 'light_use_persona', 'self'):
        text += '_self_persona ' + d['self_agent']['persona'] + '\n'
    if use_feat(opt, 'light_use_persona', 'other'):
        text += '_other_persona ' + d['partner_agent']['persona'] + '\n'
    if opt['light_use_objects']:
        for o, od in d['all_descriptions'].items():
            text += '_object_desc ' + o + ' : ' + od + '\n'
        if False:
            for o in d['room_objects'][0]:
                text += '_object_in_room ' + o + '\n'
            for o in d['carrying'][0]:
                text += '_object_carrying ' + o + '\n'
            for o in d['wearing'][0]:
                text += '_object_wearing ' + o + '\n'
            for o in d['wielding'][0]:
                text += '_object_wielding ' + o + '\n'
    for i in range(0, l, 2):
        if i < l - 1:
            if use_feat(opt, 'light_use_speech', 'partner') and d['speech'][i] is not None:
                if opt['light_use_speech_prefix']:
                    text += '_partner_say '
                text += str(d['speech'][i]) + '\n'
            if use_feat(opt, 'light_use_action', 'partner') and d['action'][i] is not None:
                text += '_partner_act ' + str(d['action'][i]) + '\n'
            if use_feat(opt, 'light_use_emote', 'partner') and d['emote'][i] is not None:
                text += '_partner_emote ' + str(d['emote'][i]) + '\n'
            if opt.get('light_use_repeat') == 'self_last':
                if i > 0:
                    text = str(d['speech'][i - 1])
                else:
                    text = 'nothing'
            if opt.get('light_use_repeat') == 'partner_last':
                text = str(d['speech'][i])
            if opt.get('light_use_repeat') == 'both_last':
                text = ''
                if i > 0:
                    text += str(d['speech'][i - 1]) + ' '
                text += str(d['speech'][i])
            label = d[label_type][i + 1]
            used_current = False
            shown = {}
            if use_feat(opt, 'light_use_current_self_output', 'speech') and label_type != 'speech' and (d['speech'][i + 1] is not None):
                if 'remove' not in opt['light_use_current_self_output']:
                    if opt['light_use_speech_prefix']:
                        text += '_self_say '
                    text += str(d['speech'][i + 1]) + '\n'
                    shown['speech'] = True
                used_current = True
            if use_feat(opt, 'light_use_current_self_output', 'all') and label_type != 'action' and use_feat(opt, 'light_use_action', 'self') and (d['action'][i + 1] is not None):
                if 'remove' not in opt['light_use_current_self_output']:
                    text += '_self_act ' + str(d['action'][i + 1]) + '\n'
                    shown['action'] = True
                used_current = True
            if use_feat(opt, 'light_use_current_self_output', 'all') and label_type != 'emote' and use_feat(opt, 'light_use_emote', 'self') and (d['emote'][i + 1] is not None):
                if 'remove' not in opt['light_use_current_self_output']:
                    text += '_self_emote ' + str(d['emote'][i + 1]) + '\n'
                    shown['emote'] = True
                used_current = True
            if 'all_filtered' in opt['light_use_current_self_output'] and used_current is False:
                label = None
            if label is not None:
                msg = {}
                msg['text'] = text
                msg['labels'] = label
                add_negs(msg, d, i + 1, label_type, split, int(opt.get('light_use_cands', 100)), opt.get('light_use_affordances', True))
                msgs.append(msg)
                text = ''
            if use_feat(opt, 'light_use_speech', 'self') and d['speech'][i + 1] is not None and ('speech' not in shown):
                if opt['light_use_speech_prefix']:
                    text += '_self_say '
                text += str(d['speech'][i + 1]) + '\n'
            if use_feat(opt, 'light_use_action', 'self') and d['action'][i + 1] is not None and ('action' not in shown):
                text += '_self_act ' + str(d['action'][i + 1]) + '\n'
            if use_feat(opt, 'light_use_emote', 'self') and d['emote'][i + 1] is not None and ('emote' not in shown):
                text += '_self_emote ' + str(d['emote'][i + 1]) + '\n'
    if len(msgs) > 0:
        msgs[-1]['episode_done'] = True
        for m in msgs:
            m['text'] = m['text'].rstrip('\n')
            fix_labels(m, opt)
            global mx
            mx = max(len(m['label_candidates']), mx)
            txt = msg_to_str(m)
            fw.write(txt + '\n')