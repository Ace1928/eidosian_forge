from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_combine_conflicting_retrieve(self):
    one = Resource.opaque(contents={})
    two = ID_AND_CHILDREN.create_resource({'foo': 'bar'})
    three = ID_AND_CHILDREN.create_resource({'baz': 'quux'})

    def foo_retrieve(uri):
        pass

    def bar_retrieve(uri):
        pass
    first = Registry(retrieve=foo_retrieve).with_resource('http://example.com/1', one)
    second = Registry().with_resource('http://example.com/2', two)
    third = Registry(retrieve=bar_retrieve).with_resource('http://example.com/3', three)
    with pytest.raises(Exception, match='conflict.*retriev'):
        first.combine(second, third)