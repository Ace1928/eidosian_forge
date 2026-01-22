import builtins as builtin_mod
import sys
import io as _io
import tokenize
from traitlets.config.configurable import Configurable
from traitlets import Instance, Float
from warnings import warn
def update_user_ns(self, result):
    """Update user_ns with various things like _, __, _1, etc."""
    if self.cache_size and result is not self.shell.user_ns['_oh']:
        if len(self.shell.user_ns['_oh']) >= self.cache_size and self.do_full_cache:
            self.cull_cache()
        update_unders = True
        for unders in ['_' * i for i in range(1, 4)]:
            if not unders in self.shell.user_ns:
                continue
            if getattr(self, unders) is not self.shell.user_ns.get(unders):
                update_unders = False
        self.___ = self.__
        self.__ = self._
        self._ = result
        if '_' not in builtin_mod.__dict__ and update_unders:
            self.shell.push({'_': self._, '__': self.__, '___': self.___}, interactive=False)
        to_main = {}
        if self.do_full_cache:
            new_result = '_%s' % self.prompt_count
            to_main[new_result] = result
            self.shell.push(to_main, interactive=False)
            self.shell.user_ns['_oh'][self.prompt_count] = result