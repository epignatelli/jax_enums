# JAX_ENUMS: Jax-compatible enumerables

[![CI](https://github.com/epignatelli/jax_enums/actions/workflows/CI.yml/badge.svg)](https://github.com/epignatelli/jax_enums/actions/workflows/CI.yml)
[![CD](https://github.com/epignatelli/jax_enums/actions/workflows/CD.yml/badge.svg)](https://github.com/epignatelli/jax_enums/actions/workflows/CD.yml)
![PyPI version](https://img.shields.io/pypi/v/jax_enums?label=PyPI&color=%230099ab)

**[Installation](#installation)** | **[Examples](#example)** | **[Cite](#cite)**


## Installation

```sh
pip install jax_enums
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
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/epignatelli/jax_enums}}
  }
```
