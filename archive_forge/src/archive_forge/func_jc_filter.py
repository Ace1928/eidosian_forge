from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError, AnsibleFilterError
import importlib
def jc_filter(data, parser, quiet=True, raw=False):
    """Convert returned command output to JSON using the JC library

    Arguments:

        parser      required    (string) the correct parser for the input data (e.g. 'ifconfig')
                                see https://github.com/kellyjonbrazil/jc#parsers for latest list of parsers.
        quiet       optional    (bool) True to suppress warning messages (default is True)
        raw         optional    (bool) True to return pre-processed JSON (default is False)

    Returns:

        dictionary or list of dictionaries

    Example:
        - name: run date command
          hosts: ubuntu
          tasks:
          - name: install the prereqs of the jc filter (jc Python package) on the Ansible controller
            delegate_to: localhost
            ansible.builtin.pip:
              name: jc
              state: present
          - ansible.builtin.shell: date
            register: result
          - ansible.builtin.set_fact:
              myvar: "{{ result.stdout | community.general.jc('date') }}"
          - ansible.builtin.debug:
              msg: "{{ myvar }}"

        produces:

        ok: [192.168.1.239] => {
            "msg": {
                "day": 9,
                "hour": 22,
                "minute": 6,
                "month": "Aug",
                "month_num": 8,
                "second": 22,
                "timezone": "UTC",
                "weekday": "Sun",
                "weekday_num": 1,
                "year": 2020
            }
        }
    """
    if not HAS_LIB:
        raise AnsibleError('You need to install "jc" as a Python library on the Ansible controller prior to running jc filter')
    try:
        if hasattr(jc, 'parse'):
            return jc.parse(parser, data, quiet=quiet, raw=raw)
        else:
            jc_parser = importlib.import_module('jc.parsers.' + parser)
            return jc_parser.parse(data, quiet=quiet, raw=raw)
    except Exception as e:
        raise AnsibleFilterError('Error in jc filter plugin:  %s' % e)