import gzip
import io
import tarfile
import sys
import os.path
from pathlib import Path
from debian.arfile import ArFile, ArError, ArMember     # pylint: disable=unused-import
from debian.changelog import Changelog
from debian.deb822 import Deb822
def tgz(self):
    """Return a TarFile object corresponding to this part of a .deb
        package.

        Despite the name, this method gives access to various kind of
        compressed tar archives, not only gzipped ones.
        """

    def _custom_decompress(command_list):
        try:
            import subprocess
            import signal
            proc = subprocess.Popen(command_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=False, preexec_fn=lambda: signal.signal(signal.SIGPIPE, signal.SIG_DFL))
        except (OSError, ValueError) as e:
            raise DebError("error while running command '%s' as subprocess: '%s'" % (' '.join(command_list), e))
        data = proc.communicate(self.__member.read())[0]
        if proc.returncode != 0:
            raise DebError("command '%s' has failed with code '%s'" % (' '.join(command_list), proc.returncode))
        return io.BytesIO(data)
    if self.__tgz is None:
        name = self.__member.name
        extension = os.path.splitext(name)[1][1:]
        if extension in PART_EXTS or name == DATA_PART or name == CTRL_PART:
            if extension == 'zst':
                buffer = _custom_decompress(['unzstd', '--stdout'])
            else:
                buffer = self.__member
            try:
                self.__tgz = tarfile.open(fileobj=buffer, mode='r:*')
            except (tarfile.ReadError, tarfile.CompressionError) as e:
                raise DebError("tarfile has returned an error: '%s'" % e)
        else:
            raise DebError("part '%s' has unexpected extension" % name)
    return self.__tgz