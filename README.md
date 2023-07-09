# JAX_ENUMS
A Jax-compatible enumerable.


## Example
```python
class Foo(Enumerable):
    BAR = 0
    BAZ = 1

def f(array: jax.Array, enumerable: Enum) -> jax.Array:
    return array[enumerable.value]

array = jnp.zeros((2, 2))
enumerable = Foo.BAR

f(array, enumerable)
jax.jit(f)(array, enumerable)
```
