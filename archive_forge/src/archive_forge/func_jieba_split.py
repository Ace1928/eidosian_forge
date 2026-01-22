from typing import List
from tokenizers import NormalizedString, PreTokenizedString, normalizers
def jieba_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
    splits = []
    for token, start, end in self.jieba.tokenize(str(normalized_string), hmm=False):
        if token in self.vocab:
            splits.append(normalized_string[start:end])
        else:
            token_list = self.normalizers.normalize_str(token).split()
            for token in token_list:
                if token:
                    end = start + len(token)
                    splits.append(normalized_string[start:end])
                    start = end
    return splits