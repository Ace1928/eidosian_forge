import pickle
import re
from debian.deprecation import function_deprecated_by
def ideal_tagset(self, tags):
    """
        Return an ideal selection of the top tags in a list of tags.

        Return the tagset made of the highest number of tags taken in
        consecutive sequence from the beginning of the given vector,
        that would intersect with the tagset of a comfortable amount
        of packages.

        Comfortable is defined in terms of how far it is from 7.
        """

    def score_fun(x):
        return float((x - 15) * (x - 15)) / x
    tagset = set()
    min_score = 3.0
    for i in range(len(tags)):
        pkgs = self.packages_of_tags(tags[:i + 1])
        card = len(pkgs)
        if card == 0:
            break
        score = score_fun(card)
        if score < min_score:
            min_score = score
            tagset = set(tags[:i + 1])
    if not tagset:
        return set(tags[:1])
    return tagset