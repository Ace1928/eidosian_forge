import collections
def unbeta(beta_metadata):
    if beta_metadata is None:
        return ()
    else:
        return tuple((_metadatum(beta_key, beta_value) for beta_key, beta_value in beta_metadata))