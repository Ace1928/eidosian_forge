import re
import torch._C as C
def registrations(self):
    output = self._format_header('Registered Kernels')
    state = self.rawRegistrations()
    state_entries = state.split('\n')
    for line in state_entries:
        first = line.split(':')[0]
        if any((first.startswith(k) for k in self.supported_keys)):
            kernel = line.split('::')[0].split(' ')[1]
            output += self._format_line(first, kernel)
    return output