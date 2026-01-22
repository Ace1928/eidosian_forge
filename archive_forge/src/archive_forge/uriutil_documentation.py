from .schema import rest_translation
return the relative path of the file in the given URI

    for uri = '/.../files/a/b/c', return 'a/b/c'

    raises ValueError (through .index()) if '/files/' is not in the URI
    