from functools import lru_cache
import json
import pytest
from referencing import Registry, Resource, exceptions
from referencing.jsonschema import DRAFT202012
from referencing.retrieval import to_cached_resource
def test_it_caches_retrieved_resources(self):
    contents = {'$schema': 'https://json-schema.org/draft/2020-12/schema'}
    stack = [json.dumps(contents)]

    @to_cached_resource()
    def retrieve(uri):
        return stack.pop()
    registry = Registry(retrieve=retrieve)
    expected = Resource.from_contents(contents)
    got = registry.get_or_retrieve('urn:example:schema')
    assert got.value == expected
    again = registry.get_or_retrieve('urn:example:schema')
    assert again.value is got.value