from __future__ import unicode_literals
import base64
import uuid
import xml.etree.ElementTree as ET
import xmltodict
from six import text_type
from winrm.transport import Transport
from winrm.exceptions import WinRMError, WinRMTransportError, WinRMOperationTimeoutError
def cleanup_command(self, shell_id, command_id):
    """
        Clean-up after a command. @see #run_command
        @param string shell_id: The shell id on the remote machine.
         See #open_shell
        @param string command_id: The command id on the remote machine.
         See #run_command
        @returns: This should have more error checking but it just returns true
         for now.
        @rtype bool
        """
    message_id = uuid.uuid4()
    req = {'env:Envelope': self._get_soap_header(resource_uri='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd', action='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Signal', shell_id=shell_id, message_id=message_id)}
    signal = req['env:Envelope'].setdefault('env:Body', {}).setdefault('rsp:Signal', {})
    signal['@CommandId'] = command_id
    signal['rsp:Code'] = 'http://schemas.microsoft.com/wbem/wsman/1/windows/shell/signal/terminate'
    res = self.send_message(xmltodict.unparse(req))
    root = ET.fromstring(res)
    relates_to = next((node for node in root.findall('.//*') if node.tag.endswith('RelatesTo'))).text
    assert uuid.UUID(relates_to.replace('uuid:', '')) == message_id