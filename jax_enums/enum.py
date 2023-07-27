# Copyright 2023 The JAX_ENUM Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

from typing import Any
from enum import Enum, EnumMeta

import jax
import jax.numpy as jnp
from dataclasses import dataclass


@jax.tree_util.register_pytree_node_class
@dataclass
class EnumItem:
    name: str
    value: jax.Array
    obj_class: str

    def __str__(self):
        return f"<{self.obj_class}.{self.name}: {self.value}> as PyTreeNode"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(tuple(jax.tree_util.tree_leaves(self)))

    def __getitem__(self, idx):
        return jax.tree_util.tree_map(lambda x: x[idx], self)

    def __eq__(self, other):
        if not isinstance(other, EnumItem):
            raise TypeError("Cannot compare EnumItem with non-EnumItem {}".format(other))
        with jax.ensure_compile_time_eval():
            return jnp.array_equal(self.value, other.value)

    def __ne__(self, other):
        return jnp.logical_not(self.__eq__(other))

    def tree_flatten(self):
        return (self.value,), (self.name, self.obj_class)

    @classmethod
    def tree_unflatten(cls, aux, children) -> EnumItem:
        return cls(value=children[0], name=aux[0], obj_class=aux[1])


class EnumerableMeta(EnumMeta):
    def __new__(mcls, name, bases, attrs):
        # this hack is to pass the equality tests in Enum.__new__
        # since, if the value of an enum is an Array,
        #  Array == Array returns an Array, which cannot be cast to bool
        for name, value in attrs.items():
            if not name.startswith("_") and not isinstance(value, EnumItem):
                # value is the value assigned to the item, e.g., 0 for A = 0
                # A == attrs["__qualname__"]
                # 0 == value
                value = EnumItem(value=jnp.asarray(value), name=name, obj_class=attrs["__qualname__"])
                dict.update(attrs, {name: value})
        return super().__new__(mcls, name, bases, attrs)

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("_") and not isinstance(value, EnumItem) and name == value.name:
            value = EnumItem(value=jnp.asarray(value.value.value), name=name, obj_class=value.value.obj_class)
        return super().__setattr__(name, value)


class Enumerable(Enum, metaclass=EnumerableMeta):
    ...
