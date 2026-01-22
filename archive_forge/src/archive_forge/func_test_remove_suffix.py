import sys
import pytest
@pytest.mark.parametrize('string, suffix, expected', (('wildcat', 'cat', 'wild'), ('blackbird', 'bird', 'black'), ('housefly', 'fly', 'house'), ('ladybug', 'bug', 'lady'), ('rattlesnake', 'snake', 'rattle'), ('seahorse', 'horse', 'sea'), ('baboon', 'badger', 'baboon'), ('quetzal', 'elk', 'quetzal')))
def test_remove_suffix(string, suffix, expected):
    result = removesuffix(string, suffix)
    assert result == expected