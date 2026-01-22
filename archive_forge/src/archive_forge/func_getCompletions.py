from __future__ import print_function
import os
import sys
import codeop
import traceback
from IPython.core.error import UsageError
from IPython.core.completer import IPCompleter
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.core.usage import default_banner_parts
from IPython.utils.strdispatch import StrDispatch
import IPython.core.release as IPythonRelease
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.core import release
from _pydev_bundle.pydev_imports import xmlrpclib
def getCompletions(self, text, act_tok):
    try:
        TYPE_IPYTHON = '11'
        TYPE_IPYTHON_MAGIC = '12'
        _line, ipython_completions = self.complete(text)
        from _pydev_bundle._pydev_completer import Completer
        completer = Completer(self.get_namespace(), None)
        ret = completer.complete(act_tok)
        append = ret.append
        ip = self.ipython
        pydev_completions = set([f[0] for f in ret])
        for ipython_completion in ipython_completions:
            if ipython_completion not in pydev_completions:
                pydev_completions.add(ipython_completion)
                inf = ip.object_inspect(ipython_completion)
                if inf['type_name'] == 'Magic function':
                    pydev_type = TYPE_IPYTHON_MAGIC
                else:
                    pydev_type = TYPE_IPYTHON
                pydev_doc = inf['docstring']
                if pydev_doc is None:
                    pydev_doc = ''
                append((ipython_completion, pydev_doc, '', pydev_type))
        return ret
    except:
        import traceback
        traceback.print_exc()
        return []