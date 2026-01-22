import collections
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine.resources import signal_responder
def merge_signal_metadata(signal_data, latest_rsrc_metadata):
    signal_data = self.normalise_signal_data(signal_data, latest_rsrc_metadata)
    if not self._metadata_format_ok(signal_data):
        LOG.info('Metadata failed validation for %s', self.name)
        raise ValueError(_('Metadata format invalid'))
    new_entry = signal_data.copy()
    unique_id = str(new_entry.pop(self.UNIQUE_ID))
    new_rsrc_metadata = latest_rsrc_metadata.copy()
    if unique_id in new_rsrc_metadata:
        LOG.info('Overwriting Metadata item for id %s!', unique_id)
    new_rsrc_metadata.update({unique_id: new_entry})
    write_attempts.append(signal_data)
    return new_rsrc_metadata