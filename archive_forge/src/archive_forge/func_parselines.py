import os
def parselines(self, lines, filename=None):
    for lineindex, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        command = parts[0]
        if command in self.commands:
            cmd = self.commands[command](*parts[1:])
        else:
            cmd = self.commands['*'](*parts)
        self.paths[cmd.path] = cmd