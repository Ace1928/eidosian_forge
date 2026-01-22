import atexit
import datetime
import re
import sqlite3
import threading
from pathlib import Path
from decorator import decorator
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from IPython.paths import locate_profile
from IPython.utils.decorators import undoc
@decorator
def catch_corrupt_db(f, self, *a, **kw):
    """A decorator which wraps HistoryAccessor method calls to catch errors from
    a corrupt SQLite database, move the old database out of the way, and create
    a new one.

    We avoid clobbering larger databases because this may be triggered due to filesystem issues,
    not just a corrupt file.
    """
    try:
        return f(self, *a, **kw)
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
        self._corrupt_db_counter += 1
        self.log.error('Failed to open SQLite history %s (%s).', self.hist_file, e)
        if self.hist_file != ':memory:':
            if self._corrupt_db_counter > self._corrupt_db_limit:
                self.hist_file = ':memory:'
                self.log.error('Failed to load history too many times, history will not be saved.')
            elif self.hist_file.is_file():
                base = str(self.hist_file.parent / self.hist_file.stem)
                ext = self.hist_file.suffix
                size = self.hist_file.stat().st_size
                if size >= _SAVE_DB_SIZE:
                    now = datetime.datetime.now().isoformat().replace(':', '.')
                    newpath = base + '-corrupt-' + now + ext
                    for i in range(100):
                        if not Path(newpath).exists():
                            break
                        else:
                            newpath = base + '-corrupt-' + now + u'-%i' % i + ext
                else:
                    newpath = base + '-corrupt' + ext
                self.hist_file.rename(newpath)
                self.log.error('History file was moved to %s and a new file created.', newpath)
            self.init_db()
            return []
        else:
            raise