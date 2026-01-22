def rekey(self):
    self.initialize_key(self._cipher.rekey(self._key))