import re
import torch._C as C
def dispatchTable(self):
    output = self._format_header('Computed Dispatch Table')
    table = self.rawDispatchTable()
    table_entries = table.split('\n')
    regex = re.compile('registered at .*FallbackKernel\\.cpp.*(\\[)')
    for line in table_entries:
        k = line.split(':')[0]
        if k in self.runtime_keys:
            entry = regex.sub('[', line)
            output += self._format_line(k, entry.split(': ')[1])
    return output