from keras.src.backend.common import global_state
Creates a sub-namespace for variable paths.

    Args:
        name: Name of the current scope (string).
        caller: Optional ID of a caller object (e.g. class instance).
        deduplicate: If `True`, if `caller` was passed,
            and the previous caller matches the current caller,
            and the previous name matches the current name,
            do not reenter a new namespace.
    