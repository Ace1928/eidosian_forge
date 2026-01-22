def dialect_of(self, other, ignore_wildcard=True):
    """Is this language a dialect (or subset/specialization) of another.

        This method returns True if this language is the same as or a
        specialization (dialect) of the other language_tag.

        If ignore_wildcard is False, then all languages will be
        considered to be a dialect of the special language tag of "*".

        """
    if not ignore_wildcard and self.is_universal_wildcard():
        return True
    for i in range(min(len(self), len(other))):
        if self.parts[i] != other.parts[i]:
            return False
    if len(self) >= len(other):
        return True
    return False