from typing import Callable
from cify.algorithm import Algorithm
from cify.objective_function import ObjectiveFunction
from cify.pso.utils import global_best, particles
from cify.pso.velocity import std_velocity


__all__ = ["PSO"]


class PSO(Algorithm):
    def __init__(
        self,
        n_particles: int,
        obj_func: ObjectiveFunction,
        velocity: Callable = std_velocity,
        velocity_params: dict = {},
    ):
        """
        :param n_particles: The number of particles in the Swarm.
        :param obj_func: The :class:`ObjectiveFunction` used to initialize the
            particles.
        """
        super().__init__()
        self.particles = particles(n_particles, obj_func)
        self.velocity = velocity
        self.velocity_params = velocity_params

    def iterate(self, obj_func) -> None:
        gb = global_best(self.particles, obj_func)
        for particle in self.particles:
            velocity = self.velocity(particle, gb, **self.velocity_params)
            particle.velocity = velocity
            particle.position = (particle.position + velocity).vector
            particle.evaluate(obj_func)
            particle.update_personal_best()

    def collection(self):
        return self.particles
