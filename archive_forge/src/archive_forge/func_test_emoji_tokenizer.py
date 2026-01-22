from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_emoji_tokenizer(self):
    """
        Test a string that contains Emoji ZWJ Sequences and skin tone modifier
        """
    tokenizer = TweetTokenizer()
    test1 = '👨\u200d👩\u200d👧\u200d👧'
    expected = ['👨\u200d👩\u200d👧\u200d👧']
    result = tokenizer.tokenize(test1)
    assert result == expected
    test2 = '👨🏿'
    expected = ['👨🏿']
    result = tokenizer.tokenize(test2)
    assert result == expected
    test3 = '🤔 🙈 me así, se😌 ds 💕👭👙 hello 👩🏾\u200d🎓 emoji hello 👨\u200d👩\u200d👦\u200d👦 how are 😊 you today🙅🏽🙅🏽'
    expected = ['🤔', '🙈', 'me', 'así', ',', 'se', '😌', 'ds', '💕', '👭', '👙', 'hello', '👩🏾\u200d🎓', 'emoji', 'hello', '👨\u200d👩\u200d👦\u200d👦', 'how', 'are', '😊', 'you', 'today', '🙅🏽', '🙅🏽']
    result = tokenizer.tokenize(test3)
    assert result == expected
    test4 = '🇦🇵🇵🇱🇪'
    expected = ['🇦🇵', '🇵🇱', '🇪']
    result = tokenizer.tokenize(test4)
    assert result == expected
    test5 = 'Hi 🇨🇦, 😍!!'
    expected = ['Hi', '🇨🇦', ',', '😍', '!', '!']
    result = tokenizer.tokenize(test5)
    assert result == expected
    test6 = '<3 🇨🇦 🤝 🇵🇱 <3'
    expected = ['<3', '🇨🇦', '🤝', '🇵🇱', '<3']
    result = tokenizer.tokenize(test6)
    assert result == expected