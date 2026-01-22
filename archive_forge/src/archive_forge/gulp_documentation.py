import os
import re
import numpy as np
from ase.units import eV, Ang
from ase.calculators.calculator import FileIOCalculator, ReadError
for t in range(3-len(g)):
                        g.append(' ')
                    for j in range(2):
                        min_index=[i+1 for i,e in enumerate(g[j][1:]) if e == '-']
                        if j==0 and len(min_index) != 0:
                            if len(min_index)==1:
                                g[2]=g[1]
                                g[1]=g[0][min_index[0]:]
                                g[0]=g[0][:min_index[0]]
                            else:
                                g[2]=g[0][min_index[1]:]
                                g[1]=g[0][min_index[0]:min_index[1]]
                                g[0]=g[0][:min_index[0]]
                                break
                        if j==1 and len(min_index) != 0:
                            g[2]=g[1][min_index[0]:]
                            g[1]=g[1][:min_index[0]]