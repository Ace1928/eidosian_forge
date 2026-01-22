from ..snap.t3mlite import simplex
from ..hyperboloid import *
def output_linked(x, tets_set):
    y = x
    while True:
        print(print_cell(y.endpoints[0].subsimplex) + '-----' + print_cell(y.endpoints[1].subsimplex), end=' ')
        y = y.next_
        if x is y:
            break
    print()
    y = x
    while True:
        print('%2d---%2d' % (y.endpoints[0].subsimplex, y.endpoints[1].subsimplex), end=' ')
        y = y.next_
        if x is y:
            break
    print()
    y = x
    while True:
        if y.tet in tets_set:
            print('   *   ', end=' ')
        else:
            print('       ', end=' ')
        y = y.next_
        if x is y:
            break
    print()
    print()