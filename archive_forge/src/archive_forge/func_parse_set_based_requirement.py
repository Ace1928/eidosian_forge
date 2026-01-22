import re
def parse_set_based_requirement(self, data):
    m = re.match('( *)([a-z0-9A-Z][a-z0-9A-Z\\._-]*[a-z0-9A-Z])( +)(notin|in)( +)\\((.*)\\)( *)', data)
    if m:
        self._set_based_requirement = True
        self._key = m.group(2)
        self._operator = m.group(4)
        self._data = [x.replace(' ', '') for x in m.group(6).split(',') if x != '']
        return True
    elif all((x not in data for x in self.equality_based_operators)):
        self._key = data.rstrip(' ').lstrip(' ')
        if self._key.startswith('!'):
            self._key = self._key[1:].lstrip(' ')
            self._operator = '!'
        return True
    return False