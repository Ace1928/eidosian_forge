import warnings
from . import exceptions as exc
from . import misc
from . import normalizers
from . import validators
def resolve_with(self, base_uri, strict=False):
    """Use an absolute URI Reference to resolve this relative reference.

        Assuming this is a relative reference that you would like to resolve,
        use the provided base URI to resolve it.

        See http://tools.ietf.org/html/rfc3986#section-5 for more information.

        :param base_uri: Either a string or URIReference. It must be an
            absolute URI or it will raise an exception.
        :returns: A new URIReference which is the result of resolving this
            reference using ``base_uri``.
        :rtype: :class:`URIReference`
        :raises rfc3986.exceptions.ResolutionError:
            If the ``base_uri`` is not an absolute URI.
        """
    if not isinstance(base_uri, URIMixin):
        base_uri = type(self).from_string(base_uri)
    if not base_uri.is_absolute():
        raise exc.ResolutionError(base_uri)
    base_uri = base_uri.normalize()
    resolving = self
    if not strict and resolving.scheme == base_uri.scheme:
        resolving = resolving.copy_with(scheme=None)
    if resolving.scheme is not None:
        target = resolving.copy_with(path=normalizers.normalize_path(resolving.path))
    elif resolving.authority is not None:
        target = resolving.copy_with(scheme=base_uri.scheme, path=normalizers.normalize_path(resolving.path))
    elif resolving.path is None:
        if resolving.query is not None:
            query = resolving.query
        else:
            query = base_uri.query
        target = resolving.copy_with(scheme=base_uri.scheme, authority=base_uri.authority, path=base_uri.path, query=query)
    else:
        if resolving.path.startswith('/'):
            path = normalizers.normalize_path(resolving.path)
        else:
            path = normalizers.normalize_path(misc.merge_paths(base_uri, resolving.path))
        target = resolving.copy_with(scheme=base_uri.scheme, authority=base_uri.authority, path=path, query=resolving.query)
    return target