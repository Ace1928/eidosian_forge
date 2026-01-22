from boto.dynamodb.batch import BatchList
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb import exceptions as dynamodb_exceptions
import time
def new_item(self, hash_key=None, range_key=None, attrs=None, item_class=Item):
    """
        Return an new, unsaved Item which can later be PUT to
        Amazon DynamoDB.

        This method has explicit (but optional) parameters for
        the hash_key and range_key values of the item.  You can use
        these explicit parameters when calling the method, such as::

            >>> my_item = my_table.new_item(hash_key='a', range_key=1,
                                        attrs={'key1': 'val1', 'key2': 'val2'})
            >>> my_item
            {u'bar': 1, u'foo': 'a', 'key1': 'val1', 'key2': 'val2'}

        Or, if you prefer, you can simply put the hash_key and range_key
        in the attrs dictionary itself, like this::

            >>> attrs = {'foo': 'a', 'bar': 1, 'key1': 'val1', 'key2': 'val2'}
            >>> my_item = my_table.new_item(attrs=attrs)
            >>> my_item
            {u'bar': 1, u'foo': 'a', 'key1': 'val1', 'key2': 'val2'}

        The effect is the same.

        .. note:
           The explicit parameters take priority over the values in
           the attrs dict.  So, if you have a hash_key or range_key
           in the attrs dict and you also supply either or both using
           the explicit parameters, the values in the attrs will be
           ignored.

        :type hash_key: int|long|float|str|unicode|Binary
        :param hash_key: The HashKey of the new item.  The
            type of the value must match the type defined in the
            schema for the table.

        :type range_key: int|long|float|str|unicode|Binary
        :param range_key: The optional RangeKey of the new item.
            The type of the value must match the type defined in the
            schema for the table.

        :type attrs: dict
        :param attrs: A dictionary of key value pairs used to
            populate the new item.

        :type item_class: Class
        :param item_class: Allows you to override the class used
            to generate the items. This should be a subclass of
            :class:`boto.dynamodb.item.Item`
        """
    return item_class(self, hash_key, range_key, attrs)