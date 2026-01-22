from .python3_compat import iterkeys, iteritems, Mapping  #, u
def toJSON(self, **options):
    """ Serializes this Munch to JSON. Accepts the same keyword options as `json.dumps()`.

            >>> b = Munch(foo=Munch(lol=True), hello=42, ponies='are pretty!')
            >>> json.dumps(b) == b.toJSON()
            True
        """
    return json.dumps(self, **options)