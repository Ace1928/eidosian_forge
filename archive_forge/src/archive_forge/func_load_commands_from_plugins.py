import os
import pkg_resources
def load_commands_from_plugins(plugins):
    commands = {}
    for plugin in plugins:
        commands.update(pkg_resources.get_entry_map(plugin, group='paste.paster_command'))
    return commands