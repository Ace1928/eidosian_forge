import string, sys
def complete_help(self, *args):
    commands = set(self.completenames(*args))
    topics = set((a[5:] for a in self.get_names() if a.startswith('help_' + args[0])))
    return list(commands | topics)