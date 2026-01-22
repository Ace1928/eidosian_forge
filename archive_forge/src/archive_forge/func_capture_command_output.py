from __future__ import absolute_import, division, print_function
import csv
import socket
import time
from string import Template
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
def capture_command_output(self, cmd, output):
    """
        Capture the output for a command
        """
    if 'command' not in self.command_results:
        self.command_results['command'] = []
    self.command_results['command'].append(cmd)
    if 'output' not in self.command_results:
        self.command_results['output'] = []
    self.command_results['output'].append(output)