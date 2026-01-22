from .jsonutil import JsonTable
def lastname(self, login):
    """ Returns the lastname of the user.
        """
    self._intf._get_entry_point()
    return JsonTable(self._intf._get_json('%s/users' % self._intf._entry)).where(login=login)['lastname']