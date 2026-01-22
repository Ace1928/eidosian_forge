import torch._C._lazy
Clear TrieCache. This is needed in testing to avoid
    node reusing between different tests.
    