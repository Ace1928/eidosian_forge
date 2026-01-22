from pyparsing import col,Word,Optional,alphas,nums
def mustMatchCols(startloc, endloc):
    return lambda s, l, t: startloc <= col(l, s) <= endloc