import os
import json
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.misc import warn_once
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, BleuMetric
import parlai.tasks.google_sgd.build as build_
def custom_evaluation(self, teacher_action: Message, labels, model_response: Message):
    resp = model_response.get('text')
    if not resp:
        return
    if teacher_action['type'] == 'apicall' and resp.startswith('apicall: '):
        gold = teacher_action['slots']
        slot_strs = resp[9:].split(' ; ')
        parsed = {}
        for slot_str in slot_strs:
            if ' = ' not in slot_str:
                if slot_str != '':
                    self.metrics.add('slot_p', AverageMetric(0))
                continue
            name, value = slot_str.split(' = ')
            parsed[name] = value
        for k, v in parsed.items():
            self.metrics.add('slot_p', AverageMetric(v == gold.get(k)))
        for k, v in gold.items():
            self.metrics.add('slot_r', AverageMetric(v == parsed.get(k)))
    elif teacher_action['type'] == 'apiresp':
        delex_resp = self._delex(resp, teacher_action['slots'])
        delex_label = self._delex(labels[0], teacher_action['slots'])
        self.metrics.add('delex_bleu', BleuMetric.compute(delex_resp, [delex_label]))