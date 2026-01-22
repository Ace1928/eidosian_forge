from __future__ import unicode_literals
import base64
import uuid
import xml.etree.ElementTree as ET
import xmltodict
from six import text_type
from winrm.transport import Transport
from winrm.exceptions import WinRMError, WinRMTransportError, WinRMOperationTimeoutError
def send_command_input(self, shell_id, command_id, stdin_input, end=False):
    """
        Send input to the given shell and command.
        @param string shell_id: The shell id on the remote machine.
         See #open_shell
        @param string command_id: The command id on the remote machine.
         See #run_command
        @param string stdin_input: The input unicode string or byte string to be sent.
        @param bool end: Boolean value which will close the stdin stream. If end=True then the stdin pipe to the
        remotely running process will be closed causing the next read by the remote process to stdin to return a
        EndOfFile error; the behavior of each process when this error is encountered is defined by the process, but most
        processes ( like CMD and powershell for instance) will just exit. Setting this value to 'True' means that no
        more input will be able to be sent to the process and attempting to do so should result in an error.
        @return: None
        """
    if isinstance(stdin_input, text_type):
        stdin_input = stdin_input.encode('437')
    req = {'env:Envelope': self._get_soap_header(resource_uri='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd', action='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Send', shell_id=shell_id)}
    stdin_envelope = req['env:Envelope'].setdefault('env:Body', {}).setdefault('rsp:Send', {}).setdefault('rsp:Stream', {})
    stdin_envelope['@CommandId'] = command_id
    stdin_envelope['@Name'] = 'stdin'
    if end:
        stdin_envelope['@End'] = 'true'
    else:
        stdin_envelope['@End'] = 'false'
    stdin_envelope['@xmlns:rsp'] = 'http://schemas.microsoft.com/wbem/wsman/1/windows/shell'
    stdin_envelope['#text'] = base64.b64encode(stdin_input)
    self.send_message(xmltodict.unparse(req))