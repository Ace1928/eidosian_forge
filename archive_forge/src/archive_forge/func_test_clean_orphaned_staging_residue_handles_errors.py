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
@mock.patch('os.listdir')
@mock.patch('os.remove')
@mock.patch.object(housekeeping, 'LOG')
def test_clean_orphaned_staging_residue_handles_errors(self, mock_LOG, mock_remove, mock_listdir):
    staging = housekeeping.staging_store_path()
    mock_listdir.return_value = [uuids.gone, uuids.error]
    mock_remove.side_effect = [FileNotFoundError('gone'), PermissionError('not yours')]
    self.cleaner.clean_orphaned_staging_residue()
    mock_LOG.error.assert_called_once_with('Failed to delete stale staging path %(path)r: %(err)s', {'path': os.path.join(staging, uuids.error), 'err': 'not yours'})
    mock_LOG.debug.assert_has_calls([mock.call('Found %i files in staging directory for potential cleanup', 2), mock.call('Stale staging residue found for image %(uuid)s: %(file)r; deleting now.', {'uuid': uuids.gone, 'file': os.path.join(staging, uuids.gone)}), mock.call('Stale staging residue found for image %(uuid)s: %(file)r; deleting now.', {'uuid': uuids.error, 'file': os.path.join(staging, uuids.error)}), mock.call('Cleaned %(cleaned)i stale staging files, %(ignored)i ignored (%(error)i errors)', {'cleaned': 1, 'ignored': 0, 'error': 1})])