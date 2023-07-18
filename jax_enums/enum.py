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

from typing import Type, Any
from enum import Enum, EnumMeta

import jax
import jax.numpy as jnp
from dataclasses import dataclass


@jax.tree_util.register_pytree_node_class
@dataclass
class EnumItem:
    name: str
    value: jax.Array
    obj_class: Type

    def __str__(self):
        return f"<{self.obj_class.__name__}.{self.name}: {self.value}> as PyTreeNode"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(tuple(jax.tree_util.tree_leaves(self)))

    def __eq__(self, other):
        if isinstance(other, EnumItem):
            with jax.ensure_compile_time_eval():
                return self.value == other.value
        return hash(self) == hash(other)

    def tree_flatten(self):
        return jnp.asarray(self.value)[None], (self.name, self.obj_class)

    @classmethod
    def tree_unflatten(cls, aux, children) -> EnumItem:
        return cls(value=children[0], name=aux[0], obj_class=aux[1])


class EnumerableMeta(EnumMeta):
    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("_"):
            value = EnumItem(name, value.value, self)
        return super().__setattr__(name, value)


class Enumerable(Enum, metaclass=EnumerableMeta):
    ...
