from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_dynamic_scope(self):
    one = ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/', 'children': [{'ID': 'child/', 'children': [{'ID': 'grandchild'}]}]})
    two = ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/two', 'children': [{'ID': 'two-child/'}]})
    registry = [one, two] @ Registry()
    resolver = registry.resolver()
    first = resolver.lookup('http://example.com/')
    second = first.resolver.lookup('#/children/0')
    third = second.resolver.lookup('grandchild')
    fourth = third.resolver.lookup('http://example.com/two')
    assert list(fourth.resolver.dynamic_scope()) == [('http://example.com/child/grandchild', fourth.resolver._registry), ('http://example.com/child/', fourth.resolver._registry), ('http://example.com/', fourth.resolver._registry)]
    assert list(third.resolver.dynamic_scope()) == [('http://example.com/child/', third.resolver._registry), ('http://example.com/', third.resolver._registry)]
    assert list(second.resolver.dynamic_scope()) == [('http://example.com/', second.resolver._registry)]
    assert list(first.resolver.dynamic_scope()) == []