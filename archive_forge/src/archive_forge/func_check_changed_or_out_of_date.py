from typing import List, Optional, Union
from . import errors, hooks, osutils, trace, tree
def check_changed_or_out_of_date(self, strict, opt_name, more_error, more_warning):
    """Check the tree for uncommitted changes and branch synchronization.

        If strict is None and not set in the config files, a warning is issued.
        If strict is True, an error is raised.
        If strict is False, no checks are done and no warning is issued.

        :param strict: True, False or None, searched in branch config if None.

        :param opt_name: strict option name to search in config file.

        :param more_error: Details about how to avoid the check.

        :param more_warning: Details about what is happening.
        """
    with self.lock_read():
        if strict is None:
            strict = self.branch.get_config_stack().get(opt_name)
        if strict is not False:
            err_class = None
            if self.has_changes():
                err_class = errors.UncommittedChanges
            elif self.last_revision() != self.branch.last_revision():
                err_class = errors.OutOfDateTree
            if err_class is not None:
                if strict is None:
                    err = err_class(self, more=more_warning)
                    trace.warning('%s', err._format())
                else:
                    err = err_class(self, more=more_error)
                    raise err