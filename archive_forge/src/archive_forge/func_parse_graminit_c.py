import re
from pgen2 import grammar, token
def parse_graminit_c(self, filename):
    """Parse the .c file written by pgen.  (Internal)

        The file looks as follows.  The first two lines are always this:

        #include "pgenheaders.h"
        #include "grammar.h"

        After that come four blocks:

        1) one or more state definitions
        2) a table defining dfas
        3) a table defining labels
        4) a struct defining the grammar

        A state definition has the following form:
        - one or more arc arrays, each of the form:
          static arc arcs_<n>_<m>[<k>] = {
                  {<i>, <j>},
                  ...
          };
        - followed by a state array, of the form:
          static state states_<s>[<t>] = {
                  {<k>, arcs_<n>_<m>},
                  ...
          };

        """
    try:
        f = open(filename)
    except OSError as err:
        print(f"Can't open {filename}: {err}")
        return False
    lineno = 0
    lineno, line = (lineno + 1, next(f))
    assert line == '#include "pgenheaders.h"\n', (lineno, line)
    lineno, line = (lineno + 1, next(f))
    assert line == '#include "grammar.h"\n', (lineno, line)
    lineno, line = (lineno + 1, next(f))
    allarcs = {}
    states = []
    while line.startswith('static arc '):
        while line.startswith('static arc '):
            mo = re.match('static arc arcs_(\\d+)_(\\d+)\\[(\\d+)\\] = {$', line)
            assert mo, (lineno, line)
            n, m, k = list(map(int, mo.groups()))
            arcs = []
            for _ in range(k):
                lineno, line = (lineno + 1, next(f))
                mo = re.match('\\s+{(\\d+), (\\d+)},$', line)
                assert mo, (lineno, line)
                i, j = list(map(int, mo.groups()))
                arcs.append((i, j))
            lineno, line = (lineno + 1, next(f))
            assert line == '};\n', (lineno, line)
            allarcs[n, m] = arcs
            lineno, line = (lineno + 1, next(f))
        mo = re.match('static state states_(\\d+)\\[(\\d+)\\] = {$', line)
        assert mo, (lineno, line)
        s, t = list(map(int, mo.groups()))
        assert s == len(states), (lineno, line)
        state = []
        for _ in range(t):
            lineno, line = (lineno + 1, next(f))
            mo = re.match('\\s+{(\\d+), arcs_(\\d+)_(\\d+)},$', line)
            assert mo, (lineno, line)
            k, n, m = list(map(int, mo.groups()))
            arcs = allarcs[n, m]
            assert k == len(arcs), (lineno, line)
            state.append(arcs)
        states.append(state)
        lineno, line = (lineno + 1, next(f))
        assert line == '};\n', (lineno, line)
        lineno, line = (lineno + 1, next(f))
    self.states = states
    dfas = {}
    mo = re.match('static dfa dfas\\[(\\d+)\\] = {$', line)
    assert mo, (lineno, line)
    ndfas = int(mo.group(1))
    for i in range(ndfas):
        lineno, line = (lineno + 1, next(f))
        mo = re.match('\\s+{(\\d+), "(\\w+)", (\\d+), (\\d+), states_(\\d+),$', line)
        assert mo, (lineno, line)
        symbol = mo.group(2)
        number, x, y, z = list(map(int, mo.group(1, 3, 4, 5)))
        assert self.symbol2number[symbol] == number, (lineno, line)
        assert self.number2symbol[number] == symbol, (lineno, line)
        assert x == 0, (lineno, line)
        state = states[z]
        assert y == len(state), (lineno, line)
        lineno, line = (lineno + 1, next(f))
        mo = re.match('\\s+("(?:\\\\\\d\\d\\d)*")},$', line)
        assert mo, (lineno, line)
        first = {}
        rawbitset = eval(mo.group(1))
        for i, c in enumerate(rawbitset):
            byte = ord(c)
            for j in range(8):
                if byte & 1 << j:
                    first[i * 8 + j] = 1
        dfas[number] = (state, first)
    lineno, line = (lineno + 1, next(f))
    assert line == '};\n', (lineno, line)
    self.dfas = dfas
    labels = []
    lineno, line = (lineno + 1, next(f))
    mo = re.match('static label labels\\[(\\d+)\\] = {$', line)
    assert mo, (lineno, line)
    nlabels = int(mo.group(1))
    for i in range(nlabels):
        lineno, line = (lineno + 1, next(f))
        mo = re.match('\\s+{(\\d+), (0|"\\w+")},$', line)
        assert mo, (lineno, line)
        x, y = mo.groups()
        x = int(x)
        if y == '0':
            y = None
        else:
            y = eval(y)
        labels.append((x, y))
    lineno, line = (lineno + 1, next(f))
    assert line == '};\n', (lineno, line)
    self.labels = labels
    lineno, line = (lineno + 1, next(f))
    assert line == 'grammar _PyParser_Grammar = {\n', (lineno, line)
    lineno, line = (lineno + 1, next(f))
    mo = re.match('\\s+(\\d+),$', line)
    assert mo, (lineno, line)
    ndfas = int(mo.group(1))
    assert ndfas == len(self.dfas)
    lineno, line = (lineno + 1, next(f))
    assert line == '\tdfas,\n', (lineno, line)
    lineno, line = (lineno + 1, next(f))
    mo = re.match('\\s+{(\\d+), labels},$', line)
    assert mo, (lineno, line)
    nlabels = int(mo.group(1))
    assert nlabels == len(self.labels), (lineno, line)
    lineno, line = (lineno + 1, next(f))
    mo = re.match('\\s+(\\d+)$', line)
    assert mo, (lineno, line)
    start = int(mo.group(1))
    assert start in self.number2symbol, (lineno, line)
    self.start = start
    lineno, line = (lineno + 1, next(f))
    assert line == '};\n', (lineno, line)
    try:
        lineno, line = (lineno + 1, next(f))
    except StopIteration:
        pass
    else:
        assert 0, (lineno, line)