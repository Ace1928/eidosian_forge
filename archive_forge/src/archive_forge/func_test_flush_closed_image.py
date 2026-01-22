from unittest import mock
from os_brick import exception
from os_brick.initiator import linuxrbd
from os_brick.tests import base
from os_brick import utils
def test_flush_closed_image(self):
    """Test when image is closed but wrapper isn't"""
    with mock.patch.object(linuxrbd, 'LOG') as mock_logger:
        self.mock_volume.image.require_not_closed.side_effect = InvalidArgument
        self.mock_volume.image.flush = mock.Mock()
        self.mock_volume_wrapper.flush()
        self.mock_volume.image.flush.assert_not_called()
        self.assertEqual(1, mock_logger.warning.call_count)
        log_msg = mock_logger.warning.call_args[0][0]
        self.assertTrue(log_msg.startswith("RBDVolumeIOWrapper's underlying image"))
        self.mock_volume.image.require_not_closed.assert_called_once_with()