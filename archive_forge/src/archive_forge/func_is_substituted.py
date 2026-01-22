from heat.common.i18n import _
def is_substituted(self, substitute_class):
    if self.substitute_class is None:
        return False
    return substitute_class is self.substitute_class