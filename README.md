# JAX_ENUMS

**[Installation](#installation)** | **[Examples](#example)** | **[Cite](#cite)**

A Jax-compatible enumerable.



## Installation
You can install `jax_enums` directly from GitHub:

```sh
pip install git+https://github.com/epignatelli/jax_enums
```

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

## Cite
```
@misc{pignatelli2023jax_enums,
  author = {Pignatelli, Eduardo},
  title = {JAX_ENUMS: JAX-compatible enumerations},
```
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/epignatelli/jax_enums}}
  }
