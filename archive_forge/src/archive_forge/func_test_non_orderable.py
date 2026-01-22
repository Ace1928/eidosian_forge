import unittest
def test_non_orderable(self):
    import warnings
    from zope.interface import ro
    try:
        del ro.__warningregistry__
    except AttributeError:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with C3Setting(ro.C3.WARN_BAD_IRO, True), C3Setting(ro.C3.STRICT_IRO, False):
            with self.assertRaises(ro.InconsistentResolutionOrderWarning):
                super().test_non_orderable()
    IOErr, _ = self._make_IOErr()
    with self.assertRaises(ro.InconsistentResolutionOrderError):
        self._callFUT(IOErr, strict=True)
    with C3Setting(ro.C3.TRACK_BAD_IRO, True), C3Setting(ro.C3.STRICT_IRO, False):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._callFUT(IOErr)
        self.assertIn(IOErr, ro.C3.BAD_IROS)
    iro = self._callFUT(IOErr, strict=False)
    legacy_iro = self._callFUT(IOErr, use_legacy_ro=True, strict=False)
    self.assertEqual(iro, legacy_iro)