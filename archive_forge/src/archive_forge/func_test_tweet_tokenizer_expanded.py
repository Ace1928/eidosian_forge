from typing import List, Tuple
import pytest
from nltk.tokenize import (
@pytest.mark.parametrize('test_input, expecteds', [('My text 0106404243030 is great text', (['My', 'text', '01064042430', '30', 'is', 'great', 'text'], ['My', 'text', '0106404243030', 'is', 'great', 'text'])), ('My ticket id is 1234543124123', (['My', 'ticket', 'id', 'is', '12345431241', '23'], ['My', 'ticket', 'id', 'is', '1234543124123'])), ('@remy: This is waaaaayyyy too much for you!!!!!! 01064042430', ([':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!', '01064042430'], [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!', '01064042430'])), ("My number is 06-46124080, except it's not.", (['My', 'number', 'is', '06-46124080', ',', 'except', "it's", 'not', '.'], ['My', 'number', 'is', '06-46124080', ',', 'except', "it's", 'not', '.'])), ("My number is 601-984-4813, except it's not.", (['My', 'number', 'is', '601-984-4813', ',', 'except', "it's", 'not', '.'], ['My', 'number', 'is', '601-984-', '4813', ',', 'except', "it's", 'not', '.'])), ("My number is (393)  928 -3010, except it's not.", (['My', 'number', 'is', '(393)  928 -3010', ',', 'except', "it's", 'not', '.'], ['My', 'number', 'is', '(', '393', ')', '928', '-', '3010', ',', 'except', "it's", 'not', '.'])), ('The product identification number is 48103284512.', (['The', 'product', 'identification', 'number', 'is', '4810328451', '2', '.'], ['The', 'product', 'identification', 'number', 'is', '48103284512', '.'])), ('My favourite substraction is 240 - 1353.', (['My', 'favourite', 'substraction', 'is', '240 - 1353', '.'], ['My', 'favourite', 'substraction', 'is', '240', '-', '1353', '.']))])
def test_tweet_tokenizer_expanded(self, test_input: str, expecteds: Tuple[List[str], List[str]]):
    """
        Test `match_phone_numbers` in TweetTokenizer.

        Note that TweetTokenizer is also passed the following for these tests:
            * strip_handles=True
            * reduce_len=True

        :param test_input: The input string to tokenize using TweetTokenizer.
        :type test_input: str
        :param expecteds: A 2-tuple of tokenized sentences. The first of the two
            tokenized is the expected output of tokenization with `match_phone_numbers=True`.
            The second of the two tokenized lists is the expected output of tokenization
            with `match_phone_numbers=False`.
        :type expecteds: Tuple[List[str], List[str]]
        """
    for match_phone_numbers, expected in zip([True, False], expecteds):
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, match_phone_numbers=match_phone_numbers)
        predicted = tokenizer.tokenize(test_input)
        assert predicted == expected