import re
from pgen2 import grammar, token
def parse_graminit_h(self, filename):
    """Parse the .h file written by pgen.  (Internal)

        This file is a sequence of #define statements defining the
        nonterminals of the grammar as numbers.  We build two tables
        mapping the numbers to names and back.

        """
    try:
        f = open(filename)
    except OSError as err:
        print(f"Can't open {filename}: {err}")
        return False
    self.symbol2number = {}
    self.number2symbol = {}
    lineno = 0
    for line in f:
        lineno += 1
        mo = re.match('^#define\\s+(\\w+)\\s+(\\d+)$', line)
        if not mo and line.strip():
            print(f"{filename}({lineno}): can't parse {line.strip()}")
        else:
            symbol, number = mo.groups()
            number = int(number)
            assert symbol not in self.symbol2number
            assert number not in self.number2symbol
            self.symbol2number[symbol] = number
            self.number2symbol[number] = symbol
    return True