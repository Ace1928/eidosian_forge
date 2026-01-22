import itertools
import os
import re
import numpy as np
def print_debug_output(results, dt):
    print('\n\n\nDETERMINISTIC TERMS: ' + dt)
    alpha = results['est']['alpha']
    print('alpha:')
    print(str(type(alpha)) + str(alpha.shape))
    print(alpha)
    print('se: ')
    print(results['se']['alpha'])
    print('t: ')
    print(results['t']['alpha'])
    print('p: ')
    print(results['p']['alpha'])
    beta = results['est']['beta']
    print('beta:')
    print(str(type(beta)) + str(beta.shape))
    print(beta)
    gamma = results['est']['Gamma']
    print('Gamma:')
    print(str(type(gamma)) + str(gamma.shape))
    print(gamma)
    if 'co' in dt or 's' in dt or 'lo' in dt:
        c = results['est']['C']
        print('C:')
        print(str(type(c)) + str(c.shape))
        print(c)
        print('se: ')
        print(results['se']['C'])