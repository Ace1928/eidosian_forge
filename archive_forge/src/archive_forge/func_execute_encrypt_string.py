from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import sys
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.vault import VaultEditor, VaultLib, match_encrypt_secret
from ansible.utils.display import Display
def execute_encrypt_string(self):
    """ encrypt the supplied string using the provided vault secret """
    b_plaintext = None
    b_plaintext_list = []
    args = [x for x in context.CLIARGS['args'] if x != '-']
    if context.CLIARGS['encrypt_string_prompt']:
        msg = 'String to encrypt: '
        name = None
        name_prompt_response = display.prompt('Variable name (enter for no name): ')
        if name_prompt_response != '':
            name = name_prompt_response
        hide_input = not context.CLIARGS['show_string_input']
        if hide_input:
            msg = 'String to encrypt (hidden): '
        else:
            msg = 'String to encrypt:'
        prompt_response = display.prompt(msg, private=hide_input)
        if prompt_response == '':
            raise AnsibleOptionsError('The plaintext provided from the prompt was empty, not encrypting')
        b_plaintext = to_bytes(prompt_response)
        b_plaintext_list.append((b_plaintext, self.FROM_PROMPT, name))
    if self.encrypt_string_read_stdin:
        if sys.stdout.isatty():
            display.display('Reading plaintext input from stdin. (ctrl-d to end input, twice if your content does not already have a newline)', stderr=True)
        stdin_text = sys.stdin.read()
        if stdin_text == '':
            raise AnsibleOptionsError('stdin was empty, not encrypting')
        if sys.stdout.isatty() and (not stdin_text.endswith('\n')):
            display.display('\n')
        b_plaintext = to_bytes(stdin_text)
        name = context.CLIARGS['encrypt_string_stdin_name']
        b_plaintext_list.append((b_plaintext, self.FROM_STDIN, name))
    if context.CLIARGS.get('encrypt_string_names', False):
        name_and_text_list = list(zip(context.CLIARGS['encrypt_string_names'], args))
        if len(args) > len(name_and_text_list):
            display.display('The number of --name options do not match the number of args.', stderr=True)
            display.display('The last named variable will be "%s". The rest will not have names.' % context.CLIARGS['encrypt_string_names'][-1], stderr=True)
        for extra_arg in args[len(name_and_text_list):]:
            name_and_text_list.append((None, extra_arg))
    else:
        name_and_text_list = [(None, x) for x in args]
    for name_and_text in name_and_text_list:
        name, plaintext = name_and_text
        if plaintext == '':
            raise AnsibleOptionsError('The plaintext provided from the command line args was empty, not encrypting')
        b_plaintext = to_bytes(plaintext)
        b_plaintext_list.append((b_plaintext, self.FROM_ARGS, name))
    outputs = self._format_output_vault_strings(b_plaintext_list, vault_id=self.encrypt_vault_id)
    b_outs = []
    for output in outputs:
        err = output.get('err', None)
        out = output.get('out', '')
        if err:
            sys.stderr.write(err)
        b_outs.append(to_bytes(out))
    b_outs.append(b'')
    self.editor.write_data(b'\n'.join(b_outs), context.CLIARGS['output_file'] or '-')
    if sys.stdout.isatty():
        display.display('Encryption successful', stderr=True)