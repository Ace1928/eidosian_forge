import re
from nltk.tree.tree import Tree
def sinica_parse(s):
    """
    Parse a Sinica Treebank string and return a tree.  Trees are represented as nested brackettings,
    as shown in the following example (X represents a Chinese character):
    S(goal:NP(Head:Nep:XX)|theme:NP(Head:Nhaa:X)|quantity:Dab:X|Head:VL2:X)#0(PERIODCATEGORY)

    :return: A tree corresponding to the string representation.
    :rtype: Tree
    :param s: The string to be converted
    :type s: str
    """
    tokens = re.split('([()| ])', s)
    for i in range(len(tokens)):
        if tokens[i] == '(':
            tokens[i - 1], tokens[i] = (tokens[i], tokens[i - 1])
        elif ':' in tokens[i]:
            fields = tokens[i].split(':')
            if len(fields) == 2:
                tokens[i] = fields[1]
            else:
                tokens[i] = '(' + fields[-2] + ' ' + fields[-1] + ')'
        elif tokens[i] == '|':
            tokens[i] = ''
    treebank_string = ' '.join(tokens)
    return Tree.fromstring(treebank_string, remove_empty_top_bracketing=True)