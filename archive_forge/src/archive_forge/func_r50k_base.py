from tiktoken.load import data_gym_to_mergeable_bpe_ranks, load_tiktoken_bpe
def r50k_base():
    mergeable_ranks = load_tiktoken_bpe('https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken', expected_hash='306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930')
    return {'name': 'r50k_base', 'explicit_n_vocab': 50257, 'pat_str': "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+", 'mergeable_ranks': mergeable_ranks, 'special_tokens': {ENDOFTEXT: 50256}}