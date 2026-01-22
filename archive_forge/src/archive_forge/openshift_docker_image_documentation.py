from __future__ import (absolute_import, division, print_function)
import re

        Docker Grammar Reference
        Reference => name [ ":" tag ] [ "@" digest ]
        name => [hostname '/'] component ['/' component]*
            hostname => hostcomponent ['.' hostcomponent]* [':' port-number]
                hostcomponent => /([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])/
                port-number   => /[0-9]+/
            component        => alpha-numeric [separator alpha-numeric]*
                alpha-numeric => /[a-z0-9]+/
                separator     => /[_.]|__|[-]*/
    