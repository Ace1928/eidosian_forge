import py
import sys
def repr_unicode(self, x, level):

    def repr(u):
        if "'" not in u:
            return py.builtin._totext("'%s'") % u
        elif '"' not in u:
            return py.builtin._totext('"%s"') % u
        else:
            return py.builtin._totext("'%s'") % u.replace("'", "\\'")
    s = repr(x[:self.maxstring])
    if len(s) > self.maxstring:
        i = max(0, (self.maxstring - 3) // 2)
        j = max(0, self.maxstring - 3 - i)
        s = repr(x[:i] + x[len(x) - j:])
        s = s[:i] + '...' + s[len(s) - j:]
    return s