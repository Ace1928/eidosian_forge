from __future__ import (absolute_import, division, print_function)
def dict_kv(value, key):
    """Return a dictionary with a single key-value pair

    Example:

        - hosts: localhost
          gather_facts: false
          vars:
            myvar: myvalue
          tasks:
          - debug:
              msg: "{{ myvar | dict_kv('thatsmyvar') }}"

        produces:

        ok: [localhost] => {
            "msg": {
                "thatsmyvar": "myvalue"
            }
        }

    Example 2:

        - hosts: localhost
          gather_facts: false
          vars:
            common_config:
              type: host
              database: all
            myservers:
            - server1
            - server2
          tasks:
          - debug:
              msg: "{{ myservers | map('dict_kv', 'server') | map('combine', common_config) }}"

        produces:

        ok: [localhost] => {
            "msg": [
                {
                    "database": "all",
                    "server": "server1",
                    "type": "host"
                },
                {
                    "database": "all",
                    "server": "server2",
                    "type": "host"
                }
            ]
        }
    """
    return {key: value}