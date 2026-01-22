import errno
import os
import random
import select
import signal
import sys
import time
import traceback
from gunicorn.errors import HaltServer, AppImportError
from gunicorn.pidfile import Pidfile
from gunicorn import sock, systemd, util
from gunicorn import __version__, SERVER_SOFTWARE
def spawn_worker(self):
    self.worker_age += 1
    worker = self.worker_class(self.worker_age, self.pid, self.LISTENERS, self.app, self.timeout / 2.0, self.cfg, self.log)
    self.cfg.pre_fork(self, worker)
    pid = os.fork()
    if pid != 0:
        worker.pid = pid
        self.WORKERS[pid] = worker
        return pid
    for sibling in self.WORKERS.values():
        sibling.tmp.close()
    worker.pid = os.getpid()
    try:
        util._setproctitle('worker [%s]' % self.proc_name)
        self.log.info('Booting worker with pid: %s', worker.pid)
        self.cfg.post_fork(self, worker)
        worker.init_process()
        sys.exit(0)
    except SystemExit:
        raise
    except AppImportError as e:
        self.log.debug('Exception while loading the application', exc_info=True)
        print('%s' % e, file=sys.stderr)
        sys.stderr.flush()
        sys.exit(self.APP_LOAD_ERROR)
    except Exception:
        self.log.exception('Exception in worker process')
        if not worker.booted:
            sys.exit(self.WORKER_BOOT_ERROR)
        sys.exit(-1)
    finally:
        self.log.info('Worker exiting (pid: %s)', worker.pid)
        try:
            worker.tmp.close()
            self.cfg.worker_exit(self, worker)
        except Exception:
            self.log.warning('Exception during worker exit:\n%s', traceback.format_exc())