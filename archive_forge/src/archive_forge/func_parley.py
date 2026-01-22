from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.legacy_2018.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.legacy_2018.worlds import MTurkOnboardWorld
from parlai.core.message import Message
from parlai.utils.strings import normalize_reply
from joblib import Parallel, delayed
import numpy as np
import os
import json
import random
import time
import torch
import copy
def parley(self):
    self.turn_idx += 1
    print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))
    'If at first turn, we need to give each agent their persona'
    if self.turn_idx == 1:
        for idx, agent in enumerate(self.agents):
            persona_text = ''
            for s in self.personas[idx]:
                persona_text += '<b><span style="color:blue">{}\n</span></b>'.format(s.strip())
            control_msg = self.get_control_msg()
            control_msg['persona_text'] = persona_text
            control_msg['text'] = self.get_instruction(tag='start', agent_id=agent.id)
            agent.observe(validate(control_msg))
            if idx == 0:
                time.sleep(3)
    'If we get to the min turns, inform turker that they can end if they\n        want.\n        '
    if self.turn_idx == self.n_turn + 1:
        for idx, agent in enumerate(self.agents):
            control_msg = self.get_control_msg()
            control_msg['text'] = self.get_instruction(idx, tag='exceed_min_turns')
            control_msg['exceed_min_turns'] = True
            agent.observe(validate(control_msg))
    'Otherwise, we proceed accordingly.'
    if self.other_first and self.turn_idx == 1:
        if self.model_agent is not None:
            persona_act = {'text': '\n'.join([self.model_persona_text, '__SILENCE__']), 'episode_done': False}
            self.model_agent.observe(persona_act)
            self.bot_seen_persona = True
            model_act = copy.deepcopy(self.model_agent.act())
            model_act.force_set('text', normalize_reply(model_act['text']))
            model_act.force_set('id', 'PERSON_2')
            self.dialog.append((1, model_act.get('text')))
            _random_delay()
            self.eval_agent.observe(_strip_tensors(model_act))
        else:
            act = self.get_human_agent_act(self.other_agent)
            timeout = self.check_timeout(act)
            if timeout:
                control_msg = self.get_control_msg()
                control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                self.eval_agent.observe(validate(control_msg))
                return
            else:
                self.dialog.append((1, act.get('text')))
                act = copy.deepcopy(act)
                act.force_set('text', normalize_reply(act['text']))
                self.eval_agent.observe(act)
    act = Message(self.get_human_agent_act(self.eval_agent))
    timeout = self.check_timeout(act)
    if timeout:
        if self.model_agent is None:
            control_msg = self.get_control_msg()
            control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
            self.other_agent.observe(validate(control_msg))
        return
    if act['episode_done']:
        if self.turn_idx >= self.n_turn:
            if not self.other_first:
                self.dialog_list = ['\n'.join([self.dialog[i][1], self.dialog[i + 1][1]]) for i in range(0, len(self.dialog), 2)]
            else:
                self.dialog_list = [' \n' + self.dialog[0][1]] + ['\n'.join([self.dialog[i][1], self.dialog[i + 1][1]]) for i in range(1, len(self.dialog) - 1, 2)]
            self.parallel_eval_mode()
            self.chat_done = True
            for ag in self.agents:
                control_msg = self.get_control_msg()
                control_msg['text'] = CHAT_ENDED_MSG
                ag.observe(validate(control_msg))
        return
    self.dialog.append((0, act['text']))
    if not self.bot_seen_persona and self.model_agent is not None:
        act.force_set('text', '\n'.join([self.model_persona_text, act['text']]))
        self.bot_seen_persona = True
    if self.model_agent is not None:
        self.model_agent.observe(act)
    else:
        act = copy.deepcopy(act)
        act.force_set('text', normalize_reply(act['text']))
        self.other_agent.observe(act)
    if not self.other_first or self.turn_idx < self.n_turn:
        if self.model_agent is not None:
            _random_delay()
            act = _strip_tensors(copy.deepcopy(self.model_agent.act()))
            act.force_set('text', normalize_reply(act['text']))
            act.force_set('id', 'PERSON_2')
        else:
            act = self.get_human_agent_act(self.other_agent)
            timeout = self.check_timeout(act)
            if timeout:
                control_msg = self.get_control_msg()
                control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                self.eval_agent.observe(validate(control_msg))
                return
        self.dialog.append((1, act.get('text')))
        act = copy.deepcopy(act)
        act.force_set('text', normalize_reply(act['text']))
        self.eval_agent.observe(act)