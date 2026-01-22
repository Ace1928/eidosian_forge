from __future__ import absolute_import, division, print_function
import traceback
from binascii import Error as binascii_error
from socket import error as socket_error
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def record_exists(self):
    update = dns.update.Update(self.zone, keyring=self.keyring, keyalgorithm=self.algorithm)
    try:
        update.present(self.module.params['record'], self.module.params['type'])
    except dns.rdatatype.UnknownRdatatype as e:
        self.module.fail_json(msg='Record error: {0}'.format(to_native(e)))
    response = self.__do_update(update)
    self.dns_rc = dns.message.Message.rcode(response)
    if self.dns_rc == 0:
        if self.module.params['state'] == 'absent':
            return 1
        for entry in self.value:
            try:
                update.present(self.module.params['record'], self.module.params['type'], entry)
            except AttributeError:
                self.module.fail_json(msg='value needed when state=present')
            except dns.exception.SyntaxError:
                self.module.fail_json(msg='Invalid/malformed value')
        response = self.__do_update(update)
        self.dns_rc = dns.message.Message.rcode(response)
        if self.dns_rc == 0:
            if self.ttl_changed():
                return 2
            else:
                return 1
        else:
            return 2
    else:
        return 0