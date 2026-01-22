import logging
import re
def compare_main(self, other):
    if not isinstance(other, SemVer):
        other = make_semver(other, self.loose)
    return compare_identifiers(str(self.major), str(other.major)) or compare_identifiers(str(self.minor), str(other.minor)) or compare_identifiers(str(self.patch), str(other.patch))