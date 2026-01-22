import sys
def disable_importlib_metadata_finder(metadata):
    """
    Ensure importlib_metadata doesn't provide older, incompatible
    Distributions.

    Workaround for #3102.
    """
    try:
        import importlib_metadata
    except ImportError:
        return
    except AttributeError:
        from .warnings import SetuptoolsWarning
        SetuptoolsWarning.emit('Incompatibility problem.', '\n            `importlib-metadata` version is incompatible with `setuptools`.\n            This problem is likely to be solved by installing an updated version of\n            `importlib-metadata`.\n            ', see_url='https://github.com/python/importlib_metadata/issues/396')
        raise
    if importlib_metadata is metadata:
        return
    to_remove = [ob for ob in sys.meta_path if isinstance(ob, importlib_metadata.MetadataPathFinder)]
    for item in to_remove:
        sys.meta_path.remove(item)