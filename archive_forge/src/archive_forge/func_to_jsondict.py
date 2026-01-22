import base64
import inspect
import builtins
def to_jsondict(self, encode_string=base64.b64encode):
    """
        This method returns a JSON style dict to describe this object.

        The returned dict is compatible with json.dumps() and json.loads().

        Suppose ClassName object inherits StringifyMixin.
        For an object like the following::

            ClassName(Param1=100, Param2=200)

        this method would produce::

            { "ClassName": {"Param1": 100, "Param2": 200} }

        This method takes the following arguments.

        .. tabularcolumns:: |l|L|

        =============  =====================================================
        Argument       Description
        =============  =====================================================
        encode_string  (Optional) specify how to encode attributes which has
                       python 'str' type.
                       The default is base64.
                       This argument is used only for attributes which don't
                       have explicit type annotations in _TYPE class attribute.
        =============  =====================================================
        """
    dict_ = {}
    encode = lambda key, val: self._encode_value(key, val, encode_string)
    for k, v in obj_attrs(self):
        dict_[k] = encode(k, v)
    return {self.__class__.__name__: dict_}