import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def debug_output(self):
    if not self.debug:
        return ''
    else:
        return '\n    # Debugging code enabled using the --debug command line switch.\n    # Will dump some variables to the top portion of the terminal.\n    echo -ne \'\\e[s\\e[H\'\n    for (( i=0; i < ${#COMP_WORDS[@]}; ++i)); do\n        echo "\\$COMP_WORDS[$i]=\'${COMP_WORDS[i]}\'"$\'\\e[K\'\n    done\n    for i in COMP_CWORD COMP_LINE COMP_POINT COMP_TYPE COMP_KEY cur curOpt; do\n        echo "\\$${i}=\\"${!i}\\""$\'\\e[K\'\n    done\n    echo -ne \'---\\e[K\\e[u\'\n'