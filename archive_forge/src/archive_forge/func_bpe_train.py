import collections
from typing import Optional
import regex
import tiktoken
def bpe_train(data: str, vocab_size: int, pat_str: str, visualise: Optional[str]='colour') -> dict[bytes, int]:
    if vocab_size < 2 ** 8:
        raise ValueError('vocab_size must be at least 256, so we can encode all bytes')
    ranks = {}
    for i in range(2 ** 8):
        ranks[bytes([i])] = i
    words: list[list[bytes]] = [[bytes([b]) for b in word.encode('utf-8')] for word in regex.findall(pat_str, data)]
    while len(ranks) < vocab_size:
        stats = collections.Counter()
        for piece in words:
            for pair in zip(piece[:-1], piece[1:]):
                stats[pair] += 1
        most_common_pair = max(stats, key=lambda x: stats[x])
        token_bytes = most_common_pair[0] + most_common_pair[1]
        token = len(ranks)
        ranks[token_bytes] = token
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_common_pair:
                    new_word.append(token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[i])
            new_words.append(new_word)
        words = new_words
        if visualise:
            print(f'The current most common pair is {most_common_pair[0]} + {most_common_pair[1]}')
            print(f'So we made {token_bytes} our {len(ranks)}th token')
            if visualise in ['colour', 'color']:
                print('Now the first fifty words in our training data look like:')
                visualise_tokens([token for word in words[:50] for token in word])
            elif visualise == 'simple':
                print('Now the first twenty words in our training data look like:')
                for word in words[:20]:
                    print(word)
            print('\n')
    return ranks