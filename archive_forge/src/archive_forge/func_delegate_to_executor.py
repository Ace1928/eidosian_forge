import functools
def delegate_to_executor(*attrs):

    def cls_builder(cls):
        for attr_name in attrs:
            setattr(cls, attr_name, _make_delegate_method(attr_name))
        return cls
    return cls_builder