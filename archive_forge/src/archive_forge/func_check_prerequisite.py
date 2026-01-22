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
def check_prerequisite(self):
    log.status.Print('---- Checking network connectivity ----')
    msg = "The Network Management API is needed to check the VM's network connectivity."
    prompt = "Is it OK to enable it and check the VM's network connectivity?"
    cancel = 'Test skipped.\nTo manually test network connectivity, try reaching another device on the same network.\n'
    try:
        prompt_continue = console_io.PromptContinue(message=msg, prompt_string=prompt, cancel_on_no=True, cancel_string=cancel)
        self.skip_troubleshoot = not prompt_continue
    except OperationCancelledError:
        self.skip_troubleshoot = True
    if self.skip_troubleshoot:
        return
    enable_api.EnableService(self.project.name, NETWORK_API)
    missing_permissions = self._CheckNetworkManagementPermissions()
    if missing_permissions:
        log.status.Print('Missing the IAM permissions {0} necessary to perform the network connectivity test. To manually test network connectivity, try reaching another device on the same network.\n'.format(' '.join(missing_permissions)))
        self.skip_troubleshoot = True
        return