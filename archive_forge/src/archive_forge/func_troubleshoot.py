from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import socket
import string
import time
from dns import resolver
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console.console_io import OperationCancelledError
import six
def troubleshoot(self):
    if self.skip_troubleshoot:
        return
    self.ip_address = self._GetSourceIPAddress()
    log.status.Print('Your source IP address is {0}\n'.format(self.ip_address))
    if not self.ip_address:
        log.status.Print("Could not resolve source external IP address, can't run network connectivity test.\n")
        self.skip_troubleshoot = True
        return
    operation_name = self._RunConnectivityTest()
    while not self._IsConnectivityTestFinish(operation_name):
        time.sleep(1)
    test_result = self._GetConnectivityTestResult()
    self._PrintConciseConnectivityTestResult(test_result)
    log.status.Print(CONNECTIVITY_TEST_MESSAGE.format(self.test_id, self.project.name))
    return