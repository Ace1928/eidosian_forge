from typing import Optional, TypeVar
def for_subclass_implementers(obj: T) -> T:
    """A decorator: Only generate docs for this method in the defining class.

  Also group this method's docs with and `@abstractmethod` in the class's docs.

  No docs will generated for this class attribute in sub-classes.

  The canonical use case for this is `tf.keras.layers.Layer.call`: It's a
  public method, essential for anyone implementing a subclass, but it should
  never be called directly.

  Works on method, or other class-attributes.

  When generating docs for a class's arributes, the `__mro__` is searched and
  the attribute will be skipped if this decorator is detected on the attribute
  on any **parent** class in the `__mro__`.

  For example:

  ```
  class Parent(object):
    @for_subclass_implementers
    def method1(self):
      pass
    def method2(self):
      pass

  class Child1(Parent):
    def method1(self):
      pass
    def method2(self):
      pass

  class Child2(Parent):
    def method1(self):
      pass
    def method2(self):
      pass
  ```

  This will produce the following docs:

  ```
  /Parent.md
    # method1
    # method2
  /Child1.md
    # method2
  /Child2.md
    # method2
  ```

  Note: This is implemented by adding a hidden attribute on the object, so it
  cannot be used on objects which do not allow new attributes to be added. So
  this decorator must go *below* `@property`, `@classmethod`,
  or `@staticmethod`:

  ```
  class Example(object):
    @property
    @for_subclass_implementers
    def x(self):
      return self._x
  ```

  Args:
    obj: The class-attribute to hide from the generated docs.

  Returns:
    obj
  """
    setattr(obj, _FOR_SUBCLASS_IMPLEMENTERS, None)
    return obj