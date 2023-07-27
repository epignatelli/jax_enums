import jax
import jax.numpy as jnp
from jax_enums import Enumerable


class Foo(Enumerable):
    BAR = 0
    BAZ = 1


class Bar(Enumerable):
    FOO = jnp.asarray([1, 2, 3])
    BAZ = jnp.asarray([4, 5, 6])


def test_jit():
    def test(enum):
        return enum == Foo.BAZ

    jax.jit(test)(Foo.BAR)


def test_vmap():
    def test(first, other):
        return first == other

    jax.vmap(test)(Bar.FOO, Bar.BAZ)


def test_equality():
    def test(first, other):
        return first == other

    jax.jit(test)(Foo.BAR, Foo.BAR)
    jax.jit(test)(Foo.BAR, Foo.BAZ)

    first = Bar.FOO
    other = Bar.BAZ
    jax.vmap(test)(first, other)


if __name__ == "__main__":
    test_jit()
    test_vmap()
    test_equality()
