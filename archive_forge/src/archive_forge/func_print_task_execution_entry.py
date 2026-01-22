import logging
import os.path
from oslo_serialization import jsonutils
from osc_lib.command import command
from cliff.lister import Lister as cliff_lister
from mistralclient.commands.v2 import base
from mistralclient import utils
def print_task_execution_entry(self, t_ex, level):
    self.print_line("task '%s' [%s] %s" % (t_ex['name'], t_ex['state'], t_ex['id']), level)
    if 'retry_count' in t_ex:
        self.print_line('(retry count: %s)' % t_ex['retry_count'], level)
    if t_ex['state'] == 'ERROR':
        state_info = t_ex['state_info']
        if state_info:
            state_info = state_info[0:100].replace('\n', ' ') + '...'
            self.print_line('(error info: %s)' % state_info, level)
    if 'action_executions' in t_ex:
        for a_ex in t_ex['action_executions']:
            self.print_action_execution_entry(a_ex, level + 1)
    if 'workflow_executions' in t_ex:
        for wf_ex in t_ex['workflow_executions']:
            self.print_workflow_execution_entry(wf_ex, level + 1)