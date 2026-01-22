import base64
import os
import stat
from cryptography import fernet
from oslo_log import log
from keystone.common import utils
import keystone.conf
def load_keys(self, use_null_key=False):
    """Load keys from disk into a list.

        The first key in the list is the primary key used for encryption. All
        other keys are active secondary keys that can be used for decrypting
        tokens.

        :param use_null_key: If true, a known key containing null bytes will be
                             appended to the list of returned keys.

        """
    if not self.validate_key_repository():
        if use_null_key:
            return [NULL_KEY]
        return []
    _, keys = self._get_key_files(self.key_repository)
    if len(keys) != self.max_active_keys:
        if self.key_repository == CONF.fernet_tokens.key_repository:
            msg = 'Loaded %(count)d Fernet keys from %(dir)s, but `[fernet_tokens] max_active_keys = %(max)d`; perhaps there have not been enough key rotations to reach `max_active_keys` yet?'
            LOG.debug(msg, {'count': len(keys), 'max': self.max_active_keys, 'dir': self.key_repository})
    key_list = [keys[x] for x in sorted(keys.keys(), reverse=True)]
    if use_null_key:
        key_list.append(NULL_KEY)
    return key_list