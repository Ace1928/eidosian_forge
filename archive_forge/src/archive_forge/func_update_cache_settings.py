from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def update_cache_settings(self):
    """Update cache block size and/or flushing threshold."""
    current_configuration = self.get_current_configuration()
    block_size = self.cache_block_size if self.cache_block_size else current_configuration['cache_settings']['cache_block_size']
    threshold = self.cache_flush_threshold if self.cache_flush_threshold else current_configuration['cache_settings']['cache_flush_threshold']
    try:
        rc, cache_settings = self.request('storage-systems/%s/symbol/setSACacheParams?verboseErrorResponse=true' % self.ssid, method='POST', data={'cacheBlkSize': block_size, 'demandFlushAmount': threshold, 'demandFlushThreshold': threshold})
    except Exception as error:
        self.module.fail_json(msg='Failed to set cache settings. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))