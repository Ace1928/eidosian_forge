from referencing import Registry
from referencing.jsonschema import DRAFT202012
from rpds import HashTrieMap, HashTrieSet
from jsonschema import Draft202012Validator
def registry_data_structures():
    return (hmap.insert('foo', 'bar'), hset.insert('foo'))