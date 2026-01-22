import functools
import logging
from multiprocessing import managers
import os
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
from oslo_rootwrap import cmd
from oslo_rootwrap import jsonrpc
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
def daemon_start(config, filters):
    temp_dir = tempfile.mkdtemp(prefix='rootwrap-')
    LOG.debug('Created temporary directory %s', temp_dir)
    try:
        rwxr_xr_x = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        os.chmod(temp_dir, rwxr_xr_x)
        socket_path = os.path.join(temp_dir, 'rootwrap.sock')
        LOG.debug('Will listen on socket %s', socket_path)
        manager_cls = get_manager_class(config, filters)
        manager = manager_cls(address=socket_path)
        server = manager.get_server()
        try:
            rw_rw_rw_ = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
            os.chmod(socket_path, rw_rw_rw_)
            try:
                stdout = sys.stdout.buffer
            except AttributeError:
                stdout = sys.stdout
            stdout.write(socket_path.encode('utf-8'))
            stdout.write(b'\n')
            stdout.write(bytes(server.authkey))
            sys.stdin.close()
            sys.stdout.close()
            sys.stderr.close()
            stop = functools.partial(daemon_stop, server)
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            LOG.info('Starting rootwrap daemon main loop')
            server.serve_forever()
        finally:
            conn = server.listener
            conn.close()
            for cl_conn in conn.get_accepted():
                try:
                    cl_conn.half_close()
                except Exception:
                    LOG.debug('Failed to close connection')
            RootwrapClass.cancel_timer()
            LOG.info('Waiting for all client threads to finish.')
            for thread in threading.enumerate():
                if thread.daemon:
                    LOG.debug('Joining thread %s', thread)
                    thread.join()
    finally:
        LOG.debug('Removing temporary directory %s', temp_dir)
        shutil.rmtree(temp_dir)