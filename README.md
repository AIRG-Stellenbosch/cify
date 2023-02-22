
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)]()
[![License](https://img.shields.io/pypi/l/cify)](https://opensource.org/licenses/MIT)

<img src=docsrc/images/cify-main-logo-slogan.png alt="logo"/>

CIFY is a framework for computational intelligence algorithms written in
Python. The goal of the project is to create a framework that allows users to
easily and reliably implement nature-inspired metaheuristics. The framework
provides a set of very simple abstractions for implementing metaheuristics,
objective functions, running experiments and collecting results.

The guiding principles of the project are: 

- **Low barrier of entry**. Reading the tutorial or looking at the examples is
  all you should need to start working with CIFY.
- **Reproducibility**. Experiments with stochastic operations should be easily
  reproducible.
- **Speed**. Computational time should be kept to a minimum.
- **Tests and documentation**. All code should be thoroughly tested and
  documented.

## Quickstart

```bash
pip install cify
```

or clone the repo and install cify to your environment using

```bash
cd CIFY
pip install -e .
```

Use the following example to get started or refer to the documentation for more
details.

```python
import numpy as np

from cify import ObjectiveFunction, Optimization, Task, set_seed
from cify.ga import GA

# 1. Set internal seed for stochastic operations.
set_seed(0)

# 2. Define an objective function.
bounds = [0.0, 1.0]
dim = 10
sphere = ObjectiveFunction(lambda vector: np.sum(vector ** 2),
                          bounds,
                          dim,
                          Optimization.Min,
                          "sphere")

# 3. Create an algorithm.
ga = GA(30, sphere)

# 4. Define a performance metric
def metric(ga: GA, f: ObjectiveFunction) -> float:
   return sorted(ga.individuals)[0]

# 5. Create a task to run the algorithm.
task = Task(ga,
           sphere,
           max_iterations=1000,
           log_iterations=100,
           metrics=[("best_of_value", metric)])

# 6. Execute
task.run()
print(task.results["best_of_value"][-1])
```
