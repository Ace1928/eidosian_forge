import datetime
import decimal
import sys
from peewee import *
from playhouse.sqlite_ext import SqliteExtDatabase

Peewee integration with pysqlcipher.

Project page: https://github.com/leapcode/pysqlcipher/

**WARNING!!! EXPERIMENTAL!!!**

* Although this extention's code is short, it has not been properly
  peer-reviewed yet and may have introduced vulnerabilities.

Also note that this code relies on pysqlcipher and sqlcipher, and
the code there might have vulnerabilities as well, but since these
are widely used crypto modules, we can expect "short zero days" there.

Example usage:

     from peewee.playground.ciphersql_ext import SqlCipherDatabase
     db = SqlCipherDatabase('/path/to/my.db', passphrase="don'tuseme4real")

* `passphrase`: should be "long enough".
  Note that *length beats vocabulary* (much exponential), and even
  a lowercase-only passphrase like easytorememberyethardforotherstoguess
  packs more noise than 8 random printable characters and *can* be memorized.

When opening an existing database, passphrase should be the one used when the
database was created. If the passphrase is incorrect, an exception will only be
raised **when you access the database**.

If you need to ask for an interactive passphrase, here's example code you can
put after the `db = ...` line:

    try:  # Just access the database so that it checks the encryption.
        db.get_tables()
    # We're looking for a DatabaseError with a specific error message.
    except peewee.DatabaseError as e:
        # Check whether the message *means* "passphrase is wrong"
        if e.args[0] == 'file is encrypted or is not a database':
            raise Exception('Developer should Prompt user for passphrase '
                            'again.')
        else:
            # A different DatabaseError. Raise it.
            raise e

See a more elaborate example with this code at
https://gist.github.com/thedod/11048875
