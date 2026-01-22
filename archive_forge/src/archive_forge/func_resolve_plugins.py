import os
import pkg_resources
def resolve_plugins(plugin_list):
    found = []
    while plugin_list:
        plugin = plugin_list.pop()
        try:
            pkg_resources.require(plugin)
        except pkg_resources.DistributionNotFound as e:
            msg = '%sNot Found%s: %s (did you run python setup.py develop?)'
            if str(e) != plugin:
                e.args = (msg % (str(e) + ': ', ' for', plugin),)
            else:
                e.args = (msg % ('', '', plugin),)
            raise
        found.append(plugin)
        dist = get_distro(plugin)
        if dist.has_metadata('paster_plugins.txt'):
            data = dist.get_metadata('paster_plugins.txt')
            for add_plugin in parse_lines(data):
                if add_plugin not in found:
                    plugin_list.append(add_plugin)
    return list(map(get_distro, found))