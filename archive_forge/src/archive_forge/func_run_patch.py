import errno
import os
import sys
import tempfile
from subprocess import PIPE, Popen
from .errors import BzrError, NoDiff3
from .textfile import check_text_path
def run_patch(directory, patches, strip=0, reverse=False, dry_run=False, quiet=False, _patch_cmd='patch', target_file=None, out=None):
    args = [_patch_cmd, '-d', directory, '-s', '-p%d' % strip, '-f']
    if quiet:
        args.append('--quiet')
    if sys.platform == 'win32':
        args.append('--binary')
    if reverse:
        args.append('-R')
    if dry_run:
        if sys.platform.startswith('freebsd'):
            args.append('--check')
        else:
            args.append('--dry-run')
        stderr = PIPE
    else:
        stderr = None
    if target_file is not None:
        args.append(target_file)
    try:
        process = Popen(args, stdin=PIPE, stdout=PIPE, stderr=stderr)
    except OSError as e:
        raise PatchInvokeError(e)
    try:
        for patch in patches:
            process.stdin.write(bytes(patch))
        process.stdin.close()
    except OSError as e:
        raise PatchInvokeError(e, process.stderr.read())
    result = process.wait()
    if not dry_run:
        if out is not None:
            out.write(process.stdout.read())
        else:
            process.stdout.read()
    if result != 0:
        raise PatchFailed()
    return result