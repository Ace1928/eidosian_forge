import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def toEnglish(self, escapeLiteral=returnUnchanged):
    keyName = self.key.toEnglish(escapeLiteral)
    if self.value:
        valueName = self.value.toEnglish(escapeLiteral)
    if self.is_scalar():
        return keyName
    elif self.is_optional():
        if self.value:
            return 'optional %s-%s pair' % (keyName, valueName)
        else:
            return 'optional %s' % keyName
    else:
        if self.n_max == sys.maxsize:
            if self.n_min:
                quantity = '%s or more ' % commafy(self.n_min)
            else:
                quantity = ''
        elif self.n_min:
            quantity = '%s to %s ' % (commafy(self.n_min), commafy(self.n_max))
        else:
            quantity = 'up to %s ' % commafy(self.n_max)
        if self.value:
            return 'map of %s%s-%s pairs' % (quantity, keyName, valueName)
        else:
            plainKeyName = self.key.toEnglish(returnUnchanged).rpartition(' ')[2].lower()
            if plainKeyName == 'chassis':
                plural = keyName
            elif plainKeyName.endswith('s'):
                plural = keyName + 'es'
            else:
                plural = keyName + 's'
            return 'set of %s%s' % (quantity, plural)