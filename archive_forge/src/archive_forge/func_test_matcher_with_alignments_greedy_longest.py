import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
def test_matcher_with_alignments_greedy_longest(en_vocab):
    cases = [('aaab', 'a* b', [0, 0, 0, 1]), ('baab', 'b a* b', [0, 1, 1, 2]), ('aaab', 'a a a b', [0, 1, 2, 3]), ('aaab', 'a+ b', [0, 0, 0, 1]), ('aaba', 'a+ b a+', [0, 0, 1, 2]), ('aabaa', 'a+ b a+', [0, 0, 1, 2, 2]), ('aaba', 'a+ b a*', [0, 0, 1, 2]), ('aaaa', 'a*', [0, 0, 0, 0]), ('baab', 'b a* b b*', [0, 1, 1, 2]), ('aabb', 'a* b* a*', [0, 0, 1, 1]), ('aaab', 'a+ a+ a b', [0, 1, 2, 3]), ('aaab', 'a+ a+ a+ b', [0, 1, 2, 3]), ('aaab', 'a+ a a b', [0, 1, 2, 3]), ('aaab', 'a+ a a', [0, 1, 2]), ('aaab', 'a+ a a?', [0, 1, 2]), ('aaaa', 'a a a a a?', [0, 1, 2, 3]), ('aaab', 'a+ a b', [0, 0, 1, 2]), ('aaab', 'a+ a+ b', [0, 0, 1, 2]), ('aaab', 'a{2,} b', [0, 0, 0, 1]), ('aaab', 'a{,3} b', [0, 0, 0, 1]), ('aaab', 'a{2} b', [0, 0, 1]), ('aaab', 'a{2,3} b', [0, 0, 0, 1])]
    for string, pattern_str, result in cases:
        matcher = Matcher(en_vocab)
        doc = Doc(matcher.vocab, words=list(string))
        pattern = []
        for part in pattern_str.split():
            if part.endswith('+'):
                pattern.append({'ORTH': part[0], 'OP': '+'})
            elif part.endswith('*'):
                pattern.append({'ORTH': part[0], 'OP': '*'})
            elif part.endswith('?'):
                pattern.append({'ORTH': part[0], 'OP': '?'})
            elif part.endswith('}'):
                pattern.append({'ORTH': part[0], 'OP': part[1:]})
            else:
                pattern.append({'ORTH': part})
        matcher.add('PATTERN', [pattern], greedy='LONGEST')
        matches = matcher(doc, with_alignments=True)
        n_matches = len(matches)
        _, s, e, expected = matches[0]
        assert expected == result, (string, pattern_str, s, e, n_matches)