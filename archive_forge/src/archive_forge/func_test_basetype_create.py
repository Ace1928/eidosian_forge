from ..base import BaseOptions, BaseType
def test_basetype_create():
    MyBaseType = CustomType.create_type('MyBaseType')
    assert isinstance(MyBaseType._meta, CustomOptions)
    assert MyBaseType._meta.name == 'MyBaseType'
    assert MyBaseType._meta.description is None