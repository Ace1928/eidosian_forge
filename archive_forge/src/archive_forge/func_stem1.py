import re
from nltk.stem.api import StemmerI
def stem1(self, token):
    """
        call this function to get the first stem
        """
    try:
        if token is None:
            raise ValueError('The word could not be stemmed, because                                  it is empty !')
        self.is_verb = False
        token = self.norm(token)
        pre = self.pref(token)
        if pre is not None:
            token = pre
        fm = self.fem2masc(token)
        if fm is not None:
            return fm
        adj = self.adjective(token)
        if adj is not None:
            return adj
        token = self.suff(token)
        ps = self.plur2sing(token)
        if ps is None:
            if pre is None:
                verb = self.verb(token)
                if verb is not None:
                    self.is_verb = True
                    return verb
        else:
            return ps
        return token
    except ValueError as e:
        print(e)