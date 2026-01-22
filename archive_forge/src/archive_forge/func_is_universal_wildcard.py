def is_universal_wildcard(self):
    """Returns True if this language tag represents all possible
        languages, by using the reserved tag of "*".

        """
    return len(self.parts) == 1 and self.parts[0] == '*'