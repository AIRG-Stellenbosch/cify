from functools import singledispatchmethod
from typing import Union, List, Optional

import numpy as np

from .objective_function import ObjectiveFunction


class Position(object):
    """
    Minimal object that defines a decision vector, and it's evaluation.
    """

    @singledispatchmethod
    def __init__(
        self,
        input: Optional[Union[np.ndarray, List, ObjectiveFunction]],
    ):
        """
        :param vector: The position vector. This will always be
        converted to a Numpy array.
        :param f: The :class:`ObjectiveFunction` to use for
        :class:`Position` evaluation.
        """
        # print("XXXXXXXXXXXXXXXXXX")
        # print(type(input))
        raise TypeError(f"Unexpected type, received f{type(input)}")

    @__init__.register(np.ndarray)
    @__init__.register(list)
    def _(
        self,
        vector: Union[np.ndarray, list],
        f: ObjectiveFunction = None,
        eval_on_init: bool = True,
    ):
        is_objective_function = isinstance(f, ObjectiveFunction)

        if f and not is_objective_function:
            raise TypeError(f"value: Expected float or None, received" f"{type(f)}")

        self.__vector = None
        self.vector = vector

        self.__value = None
        if is_objective_function:
            if eval_on_init:
                self.__value = f.eval(self.vector)
            else:
                self.__value = f.opt.default()
        self.__f = f

    @__init__.register(type(None))
    def _(
        self,
        vector: Union[np.ndarray, list],
        f: ObjectiveFunction,
        eval_on_init: bool = True,
    ):
        self.__vector = vector
        self.vector = f.sample()
        self.__value = f.eval(self.vector) if eval_on_init else f.opt.default()
        self.__f = f

    @__init__.register(ObjectiveFunction)
    def _(self, f: ObjectiveFunction, eval_on_init: bool = True):
        self.__vector = None
        self.vector = f.sample()
        self.__value = f.eval(self.vector) if eval_on_init else f.opt.default()
        self.__f = f

    @property
    def vector(self) -> np.ndarray:
        """Returns the vector as a Numpy array."""
        return self.__vector

    @property
    def value(self) -> float or list:
        """Returns the current value of the :class:`Position`."""
        return self.__value

    @property
    def f(self) -> ObjectiveFunction:
        """Returns the :class:`ObjectiveFunction` of the :class:`Position`."""
        return self.__f

    @vector.setter
    def vector(self, vector: np.ndarray or list):
        """Sets the :class:`Position` s vector."""
        if isinstance(vector, np.ndarray) or isinstance(vector, list):
            if len(vector):
                self.__vector = np.array(vector)
            else:
                raise TypeError(
                    f"vector: Expected np.ndarray or list," f"received {type(vector)}."
                )
        else:
            raise TypeError(f"Expected np.ndarray or list, received" f"{type(vector)}.")
        self.__value = None

    def dim(self):
        """Returns the dimension of the decision vector."""
        return len(self.vector)

    def copy(self):
        """Returns a deep copy of the :class:`Position` object."""
        # return Position(vector=self.vector.copy(), value=self.value)
        return Position(self.vector.copy(), self.f)

    def eval(self, f):
        """
        Evaluates the :class:`Position` and sets its ``value`` field.
        """
        self.__value = f(self.vector)
        return self.value

    def __eq__(self, other) -> bool:
        if isinstance(other, Position):
            vectors_equal = np.array_equal(self.vector, other.vector)
            values_equal = self.value == other.value
            return vectors_equal and values_equal

        return self.vector == other

    def __lt__(self, other) -> bool:
        if self.value is None:
            return False

        if isinstance(other, Position):
            if other.value is None:
                return False

            return self.value < other.value

        return self.value < other

    def __le__(self, other) -> bool:
        if self.value is None:
            return False

        if isinstance(other, Position):
            if other.value is None:
                return False

            return self.value <= other.value

        return self.value <= other

    def __gt__(self, other) -> bool:
        if self.value is None:
            return False

        if isinstance(other, Position):
            if other.value is None:
                return False

            return self.value > other.value

        return self.value > other

    def __ge__(self, other) -> bool:
        if self.value is None:
            return False

        if isinstance(other, Position):
            if other.value is None:
                return False

            return self.value >= other.value

        return self.value >= other

    def __add__(self, other):
        if type(other) == Position:
            return Position(self.__vector + other.__vector)
        else:
            return Position(self.__vector + np.array(other))

    def __sub__(self, other):
        if type(other) == Position:
            return Position(self.__vector - other.__vector)
        else:
            return Position(self.__vector - np.array(other))

    def __mul__(self, other):
        if type(other) == Position:
            return Position(self.__vector * other.__vector)
        else:
            return Position(self.__vector * np.array(other))

    def __truediv__(self, other):
        if type(other) == Position:
            return Position(self.__vector / other.__vector)
        else:
            return Position(self.__vector / np.array(other))

    def __floordiv__(self, other):
        if type(other) == Position:
            return Position(self.__vector // other.__vector)
        else:
            return Position(self.__vector // np.array(other))

    def __call__(self, f):
        return self.eval(f)

    def __len__(self) -> int:
        return len(self.vector)

    def __iter__(self):
        return iter(self.vector)

    def __getitem__(self, index):
        return self.vector[index]

    def __setitem__(self, index, value):
        self.vector[index] = value

    def __str__(self) -> str:
        return f"{self.vector} -> {self.value}"
