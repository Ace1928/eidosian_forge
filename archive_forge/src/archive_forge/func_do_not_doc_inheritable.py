from typing import Optional, TypeVar
def do_not_doc_inheritable(obj: T) -> T:
    """A decorator: Do not generate docs for this method.

  This version of the decorator is "inherited" by subclasses. No docs will be
  generated for the decorated method in any subclass. Even if the sub-class
  overrides the method.

  For example, to ensure that `method1` is **never documented** use this
  decorator on the base-class:

  ```
  class Parent(object):
    @do_not_doc_inheritable
    def method1(self):
      pass
    def method2(self):
      pass

  class Child(Parent):
    def method1(self):
      pass
    def method2(self):
      pass
  ```
  This will produce the following docs:

  ```
  /Parent.md
    # method2
  /Child.md
    # method2
  ```

  When generating docs for a class's arributes, the `__mro__` is searched and
  the attribute will be skipped if this decorator is detected on the attribute
  on any class in the `__mro__`.

  Note: This is implemented by adding a hidden attribute on the object, so it
  cannot be used on objects which do not allow new attributes to be added. So
  this decorator must go *below* `@property`, `@classmethod`,
  or `@staticmethod`:

  ```
  class Example(object):
    @property
    @do_not_doc_inheritable
    def x(self):
      return self._x
  ```

  Args:
    obj: The class-attribute to hide from the generated docs.

  Returns:
    obj
  """
    setattr(obj, _DO_NOT_DOC_INHERITABLE, None)
    return obj