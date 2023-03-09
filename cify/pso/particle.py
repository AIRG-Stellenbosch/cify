import numpy as np
from typing import Union

from cify import Position, ObjectiveFunction


__all__ = ["Particle"]


class Particle(object):
    """
    Defines a particle used in PSO.
    """

    def __init__(
        self,
        position: Union[np.ndarray, list, Position] = None,
        velocity=None,
        f: ObjectiveFunction = None,
    ):
        """
        :param position: The initial position of the particle. If unspecified,
        the position will default to a uniformly sampled vector within the
        bounds of the :class:`ObjectiveFunction` (if provided).
        :param velocity: The initial velocity. Defaults to a vector of zeros.
        :param f: The objective function to be optimized.
        """
        super().__init__()

        if isinstance(position, Position):
            self.__position = position
        else:
            self.__position = Position(position, f)

        self.__personal_best = self.__position.copy()

        if velocity:
            self.__velocity = velocity
        else:
            self.__velocity = np.zeros(self.__position.dim())

    @property
    def position(self) -> Position:
        """Returns the position as a :class:`Position` object."""
        return self.__position

    @property
    def personal_best(self) -> Position:
        """Returns the personal best position as a :class:`Position` object."""
        return self.__personal_best

    @property
    def velocity(self) -> np.ndarray:
        """Returns the current velocity as a Numpy array."""
        return self.__velocity

    @position.setter
    def position(self, position):
        """Set's the :class:`Individual`'s ``position``."""
        self.__position = Position(position)

    @velocity.setter
    def velocity(self, velocity):
        """Set's the :class:`Individual`'s velocity."""
        self.__velocity = np.array(velocity)

    def update_personal_best(self):
        if self.personal_best > self.position:
            self.__personal_best = self.position.copy()

    def evaluate(self, f: ObjectiveFunction):
        """Evaluates the :class:`Particle`'s position."""
        self.__position.eval(f)

    def __eq__(self, other) -> bool:
        return self.position == other.position

    def __lt__(self, other) -> bool:
        return self.position < other.position

    def __le__(self, other) -> bool:
        return self.position <= other.position

    def __gt__(self, other) -> bool:
        return self.position > other.position

    def __ge__(self, other) -> bool:
        return self.position >= other.position

    def __str__(self) -> str:
        """Returns a str representation of the :class:`Particle`."""
        return f"position: {self.position}\n" f"velocity: {self.velocity}\n"
