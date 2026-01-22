import pkgutil
from typing import Optional
from absl import logging
def load_local_discovery_doc(doc_filename: str) -> bytes:
    """Loads the discovery document for `doc_filename` with `version` from package files.

  Example:
    bq_disc_doc = discovery_document_loader
      .load_local_discovery_doc('discovery_next/bigquery.json')

  Args:
    doc_filename: [str], The filename of the discovery document to be loaded.

  Raises:
    FileNotFoundError: If no discovery doc could be loaded.

  Returns:
    `bytes`, On success, A json object with the contents of the
    discovery document. On failure, None.
  """
    doc = _fetch_discovery_doc_from_pkg(PKG_NAME, doc_filename)
    if not doc:
        raise FileNotFoundError('Failed to load discovery doc from resource path: %s.%s' % (PKG_NAME, doc_filename))
    return doc