
    Imports a module, but catches import errors.  Only catches errors
    when that module doesn't exist; if that module itself has an
    import error it will still get raised.  Returns None if the module
    doesn't exist.
    