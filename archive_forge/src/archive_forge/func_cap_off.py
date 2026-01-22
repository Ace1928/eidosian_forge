from ..sage_helper import _within_sage
from . import exhaust
from .links_base import Link
def cap_off(self, i):
    ans_dict = {}
    for matching, coeff in self.dict.items():
        new_matching, has_circle = cap_off(matching, i)
        cur_coeff = ans_dict.get(new_matching, R.zero())
        if has_circle:
            coeff = (q + q ** (-1)) * coeff
        ans_dict[new_matching] = cur_coeff + coeff
    return VElement(ans_dict)