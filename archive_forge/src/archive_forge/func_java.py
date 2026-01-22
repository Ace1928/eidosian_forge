import fnmatch
import locale
import os
import re
import stat
import subprocess
import sys
import textwrap
import types
import warnings
from xml.etree import ElementTree
def java(cmd, classpath=None, stdin=None, stdout=None, stderr=None, blocking=True):
    """
    Execute the given java command, by opening a subprocess that calls
    Java.  If java has not yet been configured, it will be configured
    by calling ``config_java()`` with no arguments.

    :param cmd: The java command that should be called, formatted as
        a list of strings.  Typically, the first string will be the name
        of the java class; and the remaining strings will be arguments
        for that java class.
    :type cmd: list(str)

    :param classpath: A ``':'`` separated list of directories, JAR
        archives, and ZIP archives to search for class files.
    :type classpath: str

    :param stdin: Specify the executed program's
        standard input file handles, respectively.  Valid values are ``subprocess.PIPE``,
        an existing file descriptor (a positive integer), an existing
        file object, 'pipe', 'stdout', 'devnull' and None.  ``subprocess.PIPE`` indicates that a
        new pipe to the child should be created.  With None, no
        redirection will occur; the child's file handles will be
        inherited from the parent.  Additionally, stderr can be
        ``subprocess.STDOUT``, which indicates that the stderr data
        from the applications should be captured into the same file
        handle as for stdout.

    :param stdout: Specify the executed program's standard output file
        handle. See ``stdin`` for valid values.

    :param stderr: Specify the executed program's standard error file
        handle. See ``stdin`` for valid values.


    :param blocking: If ``false``, then return immediately after
        spawning the subprocess.  In this case, the return value is
        the ``Popen`` object, and not a ``(stdout, stderr)`` tuple.

    :return: If ``blocking=True``, then return a tuple ``(stdout,
        stderr)``, containing the stdout and stderr outputs generated
        by the java command if the ``stdout`` and ``stderr`` parameters
        were set to ``subprocess.PIPE``; or None otherwise.  If
        ``blocking=False``, then return a ``subprocess.Popen`` object.

    :raise OSError: If the java command returns a nonzero return code.
    """
    subprocess_output_dict = {'pipe': subprocess.PIPE, 'stdout': subprocess.STDOUT, 'devnull': subprocess.DEVNULL}
    stdin = subprocess_output_dict.get(stdin, stdin)
    stdout = subprocess_output_dict.get(stdout, stdout)
    stderr = subprocess_output_dict.get(stderr, stderr)
    if isinstance(cmd, str):
        raise TypeError('cmd should be a list of strings')
    if _java_bin is None:
        config_java()
    if isinstance(classpath, str):
        classpaths = [classpath]
    else:
        classpaths = list(classpath)
    classpath = os.path.pathsep.join(classpaths)
    cmd = list(cmd)
    cmd = ['-cp', classpath] + cmd
    cmd = [_java_bin] + _java_options + cmd
    p = subprocess.Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr)
    if not blocking:
        return p
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print(_decode_stdoutdata(stderr))
        raise OSError('Java command failed : ' + str(cmd))
    return (stdout, stderr)