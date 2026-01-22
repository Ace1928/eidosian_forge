@property
def default_template(self):
    if self.func:
        return '%(func)s(%(lhs)s, %(rhs)s)'
    else:
        return '%(lhs)s %(op)s %(rhs)s'