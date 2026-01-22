from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def get_passphrase(module):
    """
    Attempt to retrieve passphrase from keyring using the Python API and fallback to using a shell.
    """
    try:
        passphrase = keyring.get_password(module.params['service'], module.params['username'])
        return passphrase
    except keyring.errors.KeyringLocked:
        pass
    except keyring.errors.InitError:
        pass
    except AttributeError:
        pass
    get_argument = 'echo "%s" | gnome-keyring-daemon --unlock\nkeyring get %s %s\n' % (quote(module.params['keyring_password']), quote(module.params['service']), quote(module.params['username']))
    dummy, stdout, dummy = module.run_command('dbus-run-session -- /bin/bash', use_unsafe_shell=True, data=get_argument, encoding=None)
    try:
        return stdout.decode('UTF-8').splitlines()[1]
    except IndexError:
        return None