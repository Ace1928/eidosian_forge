import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def bash_completion_function(out, function_name='_brz', function_only=False, debug=False, no_plugins=False, selected_plugins=None):
    dc = DataCollector(no_plugins=no_plugins, selected_plugins=selected_plugins)
    data = dc.collect()
    cg = BashCodeGen(data, function_name=function_name, debug=debug)
    if function_only:
        res = cg.function()
    else:
        res = cg.script()
    out.write(res)