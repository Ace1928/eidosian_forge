import re
import string
def inverse_perm(L):
    ans = len(L) * [None]
    for i, x in enumerate(L):
        ans[x] = i
    return ans