from boto.sdb.db.property import Property
from boto.sdb.db.key import Key
from boto.sdb.db.query import Query
import boto
from boto.compat import filter
def delete_attributes(self, attrs):
    """
        Delete just these attributes, not the whole object.

        :param attrs: Attributes to save, as a list of string names
        :type attrs: list
        :return: self
        :rtype: :class:`boto.sdb.db.model.Model`
        """
    assert isinstance(attrs, list), 'Argument must be a list of names of keys to delete.'
    self._manager.domain.delete_attributes(self.id, attrs)
    self.reload()
    return self