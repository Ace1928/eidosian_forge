def get_pydantic_major_version() -> int:
    """Get the major version of Pydantic."""
    try:
        import pydantic
        return int(pydantic.__version__.split('.')[0])
    except ImportError:
        return 0