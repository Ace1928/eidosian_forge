import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
def test_bzr_connect_to_bzr_ssh(self):
    """get_transport of a bzr+ssh:// behaves correctly.

        bzr+ssh:// should cause bzr to run a remote bzr smart server over SSH.
        """
    self.requireFeature(features.paramiko)
    from breezy.tests import stub_sftp
    self.command_executed = []
    started = []

    class StubSSHServer(stub_sftp.StubServer):
        test = self

        def check_channel_exec_request(self, channel, command):
            self.test.command_executed.append(command)
            proc = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

            def ferry_bytes(read, write, close):
                while True:
                    bytes = read(1)
                    if bytes == b'':
                        close()
                        break
                    write(bytes)
            file_functions = [(channel.recv, proc.stdin.write, proc.stdin.close), (proc.stdout.read, channel.sendall, channel.close), (proc.stderr.read, channel.sendall_stderr, channel.close)]
            started.append(proc)
            for read, write, close in file_functions:
                t = threading.Thread(target=ferry_bytes, args=(read, write, close))
                t.start()
                started.append(t)
            return True
    ssh_server = stub_sftp.SFTPFullAbsoluteServer(StubSSHServer)
    self.start_server(ssh_server)
    port = ssh_server.port
    bzr_remote_command = self.get_brz_command()
    self.overrideEnv('BZR_REMOTE_PATH', ' '.join(bzr_remote_command))
    self.overrideEnv('PYTHONPATH', ':'.join(sys.path))
    path_to_branch = osutils.abspath('.')
    if sys.platform == 'win32':
        path_to_branch = '/' + path_to_branch
    url = 'bzr+ssh://fred:secret@localhost:%d%s' % (port, path_to_branch)
    t = transport.get_transport(url)
    self.permit_url(t.base)
    t.mkdir('foo')
    self.assertEqual([b'%s serve --inet --directory=/ --allow-writes' % ' '.join(bzr_remote_command).encode()], self.command_executed)
    t._client._medium.disconnect()
    if not started:
        return
    started[0].wait()
    for t in started[1:]:
        t.join()