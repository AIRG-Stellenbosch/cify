import operator
from enum import Enum

import numpy as np

__all__ = ["Optimization"]


class Optimization(Enum):
    """
    The :class:`Optimization` class is an enum that is used to define whether
    an :class:`ObjectiveFunction` is to be minimized or maximized. The class
    contains two methods, of which one is more useful, ``cmp``.
    """

    Min = 1
    Max = 2

    def cmp(self, a, b, equal=False) -> bool:
        """
        A comparison function returned by an objective function which can be
        used when constructing algorithms to handle optimizing minimization or
        maximization objective functions. The ``cmp`` function takes two
        arguments and returns ``True`` if ``a`` is better than ``b`` otherwise
        ``False``.

        For minimization objective functions, ``cmp`` function checks whether
        ``a`` is less than ``b``, and for maximization objective functions
        ``cmp`` checks whether ``a`` greater than ``b``. Think of it as saying,
        "is ``a`` better than the ``b``".

        :return: Whether ``a`` is better than ``b``.
        """
        if equal:
            if self.value == 1:
                return operator.le(a, b)

            return operator.ge(a, b)

        if self.value == 1:
            return operator.lt(a, b)

        return operator.gt(a, b)

    def best(self, a, b) -> bool:
        """
        Use ``cmp`` to return the better parameter.

        :return: ``a`` if ``cmp(a,b) is True otherwise ``b``.
        """
        if self.value == 1:
            return a if operator.lt(a, b) else b

        return a if operator.gt(a, b) else b

    def default(self):
        """
        :return: A worst case value, i.e ``np.inf`` for ``Min`` and ``-np.inf``
        for ``Max``.
        """
        if self.value == 1:
            return np.inf

        return -1 * np.inf

    def verb(self) -> str:
        """
        :return: A verb representation of the optimization type.
        """
        if self.value == 1:
            return "Minimizing"

        return "Maximizing"
