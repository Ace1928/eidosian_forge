from axolotl.state.prekeystore import PreKeyStore
from axolotl.state.prekeyrecord import PreKeyRecord
from yowsup.axolotl.exceptions import InvalidKeyIdException
import sys
def loadPendingPreKeys(self):
    q = 'SELECT record FROM prekeys'
    cursor = self.dbConn.cursor()
    cursor.execute(q)
    result = cursor.fetchall()
    return [PreKeyRecord(serialized=result[0]) for result in result]