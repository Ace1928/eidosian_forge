from tiktoken.load import data_gym_to_mergeable_bpe_ranks, load_tiktoken_bpe
def cl100k_base():
    mergeable_ranks = load_tiktoken_bpe('https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken', expected_hash='223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7')
    special_tokens = {ENDOFTEXT: 100257, FIM_PREFIX: 100258, FIM_MIDDLE: 100259, FIM_SUFFIX: 100260, ENDOFPROMPT: 100276}
    return {'name': 'cl100k_base', 'pat_str': "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+", 'mergeable_ranks': mergeable_ranks, 'special_tokens': special_tokens}