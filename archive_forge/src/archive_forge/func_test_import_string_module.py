from pytest import raises
from graphene import ObjectType, String
from ..module_loading import import_string, lazy_import
def test_import_string_module():
    with raises(Exception) as exc_info:
        import_string('graphenea')
    assert str(exc_info.value) == "graphenea doesn't look like a module path"