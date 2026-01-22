import os
import pkg_resources
def load_global_commands():
    commands = {}
    for p in pkg_resources.iter_entry_points('paste.global_paster_command'):
        commands[p.name] = p
    return commands