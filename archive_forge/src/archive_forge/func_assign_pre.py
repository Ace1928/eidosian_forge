from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
def assign_pre(self, token: Any) -> Any:
    token_line = token.start_mark.line
    if _debug != 0:
        import inspect
        info = inspect.getframeinfo(inspect.stack()[1][0])
        xprintf('assign_pre', token_line, self.unused, info.function, info.lineno)
    gobbled = False
    while self.unused and self.unused[0] < token_line:
        gobbled = True
        first = self.unused.pop(0)
        if _debug != 0:
            xprintf('assign_pre < ', first)
        self.comments[first].set_used()
        token.add_comment_pre(first)
    return gobbled