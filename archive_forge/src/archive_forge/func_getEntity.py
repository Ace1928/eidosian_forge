from twisted.python import reflect
def getEntity(self, name, request):
    """Retrieve an entity from me.

        I will first attempt to retrieve an entity statically; static entities
        will obscure dynamic ones.  If that fails, I will retrieve the entity
        dynamically.

        If I cannot retrieve an entity, I will return 'None'.
        """
    ent = self.getStaticEntity(name)
    if ent is not None:
        return ent
    ent = self.getDynamicEntity(name, request)
    if ent is not None:
        return ent
    return None