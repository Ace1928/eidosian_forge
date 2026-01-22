import importlib

    Helper function to import submodules lazily in Python 3.7+

    Parameters
    ----------
    rel_modules: list of str
        list of submodules to import, of the form .submodule
    rel_classes: list of str
        list of submodule classes/variables to import, of the form ._submodule.Foo

    Returns
    -------
    tuple
        Tuple that should be assigned to __all__, __getattr__ in the caller
    