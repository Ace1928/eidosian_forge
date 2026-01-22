import os
from unittest import mock
import glance_store
from oslo_config import cfg
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context
from glance import housekeeping
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch('os.remove')
@mock.patch('os.listdir')
@mock.patch.object(housekeeping, 'LOG')
def test_clean_orphaned_staging_residue(self, mock_LOG, mock_listdir, mock_remove):
    staging = housekeeping.staging_store_path()
    image = self.db.image_create(self.context, {'status': 'queued'})
    mock_listdir.return_value = ['notanimageid', image['id'], uuids.stale, uuids.midconvert, '%s.qcow2' % uuids.midconvert]
    self.cleaner.clean_orphaned_staging_residue()
    expected_stale = os.path.join(staging, uuids.stale)
    expected_mc = os.path.join(staging, uuids.midconvert)
    expected_mc_target = os.path.join(staging, '%s.qcow2' % uuids.midconvert)
    mock_remove.assert_has_calls([mock.call(expected_stale), mock.call(expected_mc), mock.call(expected_mc_target)])
    mock_LOG.debug.assert_has_calls([mock.call('Found %i files in staging directory for potential cleanup', 5), mock.call('Staging directory contains unexpected non-image file %r; ignoring', 'notanimageid'), mock.call('Stale staging residue found for image %(uuid)s: %(file)r; deleting now.', {'uuid': uuids.stale, 'file': expected_stale}), mock.call('Stale staging residue found for image %(uuid)s: %(file)r; deleting now.', {'uuid': uuids.midconvert, 'file': expected_mc}), mock.call('Stale staging residue found for image %(uuid)s: %(file)r; deleting now.', {'uuid': uuids.midconvert, 'file': expected_mc_target}), mock.call('Cleaned %(cleaned)i stale staging files, %(ignored)i ignored (%(error)i errors)', {'cleaned': 3, 'ignored': 2, 'error': 0})])