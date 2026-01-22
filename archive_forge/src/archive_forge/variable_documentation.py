import sys
Expand the variable in question.

        Using ``var_dict`` and the previously parsed defaults, expand this
        variable and subvariables.

        :param dict var_dict: dictionary of key-value pairs to be used during
            expansion
        :returns: dict(variable=value)

        Examples::

            # (1)
            v = URIVariable('/var')
            expansion = v.expand({'var': 'value'})
            print(expansion)
            # => {'/var': '/value'}

            # (2)
            v = URIVariable('?var,hello,x,y')
            expansion = v.expand({'var': 'value', 'hello': 'Hello World!',
                                  'x': '1024', 'y': '768'})
            print(expansion)
            # => {'?var,hello,x,y':
            #     '?var=value&hello=Hello%20World%21&x=1024&y=768'}

        