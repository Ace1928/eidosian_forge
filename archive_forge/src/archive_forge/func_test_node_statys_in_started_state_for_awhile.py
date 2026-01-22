import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
def test_node_statys_in_started_state_for_awhile(self):
    self.mock_operations.get_node_state.side_effect = ['started', 'started', 'stopped']
    self.assertTrue(self.destroyer.destroy_node(1))
    self.mock_operations.stop_node.assert_called_once_with(1)
    self.mock_sleep.assert_has_calls([call(self.destroyer.WAIT_AMOUNT)])
    self.mock_operations.destroy_node.assert_called_once_with(1)