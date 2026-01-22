from pyparsing import (
import pydot
def push_subgraph_stmt(s, loc, toks):
    g = pydot.Subgraph('')
    for e in toks:
        if len(e) == 3:
            e[2].set_name(e[1])
            if e[0] == 'subgraph':
                e[2].obj_dict['show_keyword'] = True
            return e[2]
        else:
            if e[0] == 'subgraph':
                e[1].obj_dict['show_keyword'] = True
            return e[1]
    return g