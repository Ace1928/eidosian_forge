from typing import List, Tuple
import pytest
from nltk.tokenize import (
@pytest.mark.parametrize('sentences, expected', [('this is a test. . new sentence.', ['this is a test.', '.', 'new sentence.']), ('This. . . That', ['This.', '.', '.', 'That']), ('This..... That', ['This..... That']), ('This... That', ['This... That']), ('This.. . That', ['This.. .', 'That']), ('This. .. That', ['This.', '.. That']), ('This. ,. That', ['This.', ',.', 'That']), ('This!!! That', ['This!!!', 'That']), ('This! That', ['This!', 'That']), ("1. This is R .\n2. This is A .\n3. That's all", ['1.', 'This is R .', '2.', 'This is A .', '3.', "That's all"]), ("1. This is R .\t2. This is A .\t3. That's all", ['1.', 'This is R .', '2.', 'This is A .', '3.', "That's all"]), ('Hello.\tThere', ['Hello.', 'There'])])
def test_sent_tokenize(self, sentences: str, expected: List[str]):
    assert sent_tokenize(sentences) == expected