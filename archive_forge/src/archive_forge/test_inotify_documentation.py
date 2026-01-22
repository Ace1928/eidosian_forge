from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase

        If C{inotify_add_watch} returns a negative number, L{add}
        raises L{INotifyError}.
        