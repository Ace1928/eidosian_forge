from typing import Optional, TypeVar
def do_not_generate_docs(obj: T) -> T:
    """A decorator: Do not generate docs for this object.

  For example the following classes:

  ```
  class Parent(object):
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

  Produce the following api_docs:

  ```
  /Parent.md
    # method1
    # method2
  /Child.md
    # method1
    # method2
  ```

  This decorator allows you to skip classes or methods:

  ```
  @do_not_generate_docs
  class Parent(object):
    def method1(self):
      pass
    def method2(self):
      pass

  class Child(Parent):
    @do_not_generate_docs
    def method1(self):
      pass
    def method2(self):
      pass
  ```

  This will only produce the following docs:

  ```
  /Child.md
    # method2
  ```

  Note: This is implemented by adding a hidden attribute on the object, so it
  cannot be used on objects which do not allow new attributes to be added. So
  this decorator must go *below* `@property`, `@classmethod`,
  or `@staticmethod`:

  ```
  class Example(object):
    @property
    @do_not_generate_docs
    def x(self):
      return self._x
  ```

  Args:
    obj: The object to hide from the generated docs.

  Returns:
    obj
  """
    setattr(obj, _DO_NOT_DOC, None)
    return obj