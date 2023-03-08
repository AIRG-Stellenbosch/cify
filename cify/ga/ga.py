from typing import Callable
import numpy as np

from cify import Algorithm, Position, ObjectiveFunction

from .crossover import uniform_crossover
from .mutate import mutate
from .select import top

__all__ = ["GA"]


class GA(Algorithm):
    def __init__(
        self,
        n: int,
        f: ObjectiveFunction,
        crossover: Callable = uniform_crossover,
        mutation: Callable = mutate,
        selection: Callable = top,
        crossover_params: dict = {"pc": 0.5},
        mutation_params: dict = {"pm": 0.5, "ms": 0.15},
        selection_params: dict = { },
    ):
        """
        :param n: The number of individuals in the population.
        
        :param pm: probability of mutation?
        :param pc: probability of crossover?
        :param ms: mutation scale?
        """
        super().__init__()
        self.individuals = [Position(f) for _ in range(n)]

        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.crossover_params = crossover_params
        self.mutation_params = mutation_params
        self.selection_params = selection_params

    def iterate(self, f: ObjectiveFunction):
        n = len(self.individuals) // 2
        elite = top(n, self.individuals, f.opt, **self.selection_params)
        next_gen = []
        for parent_a in elite:
            parent_b_idx = int(np.random.uniform(0, len(elite) - 1))
            parent_b = elite[parent_b_idx]
            child_1, child_2 = uniform_crossover(parent_a, parent_b, **self.crossover_params)
            child_1 = mutate(child_1, **self.mutation_params)
            child_2 = mutate(child_1, **self.mutation_params)
            child_1(f)
            child_2(f)
            next_gen.append(child_1)
            next_gen.append(child_2)

        self.individuals = next_gen
