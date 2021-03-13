import numpy as np
import time
from copy import deepcopy
from functools import partial, reduce
from itertools import product
from collections.abc import Mapping, Sequence, Iterable
import operator
from simulation.model.epidemic import Epidemic


class ParameterSearch:
    """Gradient-based search to estimate network characteristics and disease
    dynamics of a pandemic, based on the SIRV model"""

    def __init__(self,
                 graph_generator,
                 graph_generator_check,
                 n_nodes,
                 steps,
                 n_infected_init,
                 vacc,
                 ni_target):
        """Set the vaccination vector, the steps of the simulation and the
        newly infected individuals target.

        Args:
            graph_generator: function to generate a nx graph, used to run
                simulations
            graph_generator_check: function to check validity of parameters
                for the graph
            n_nodes: total number of individuals in the population
            steps: number of steps (e.g., weeks) of the simulation
            n_infected_init: number of infected nodes in the initial
                configuration, chosen randomly among all nodes of the graph
                according to a uniform probability distribution
            vacc: total fraction of population that has received vaccination by
                each week
            ni_target: newly infected nodes target

        """

        self.vacc = vacc
        self.steps = steps
        self.ni_target = np.array(ni_target)

        self.n_infected_init = n_infected_init

        self.generator = graph_generator
        self.generator_check = graph_generator_check
        self.n_nodes = n_nodes

    def search(self,
               graph_initial_params,
               epidemic_initial_params,
               graph_delta_params,
               epidemic_delta_params,
               graph_delta_fine_params,
               epidemic_delta_fine_params,
               simulations_per_grid):
        """Estimate the best parameters for the simulation according to the
        newly infected individuals target `ni_target`.

        Args:
            graph_initial_params (dict): initial parameters for the graph
            epidemic_initial_params (dict): initial parameters for the epidemic,
                namely beta and rho
            graph_delta_params (dict): deltas for searching the parameter space
                of the graph in the first phase
            epidemic_delta_params (dict): deltas for searching the parameter
                space of the epidemic model in the first phase
            graph_delta_fine_params (dict): deltas for searching the parameter
                space of the graph in the second phase
            epidemic_delta_fine_params (dict): deltas for searching the
                parameter space of the epidemic model in the second phase
            simulations_per_grid (int): number of simulations to run for each
                parameter set

        """

        graph_param_names = set(graph_initial_params.keys())
        epidemic_param_names = set(epidemic_initial_params.keys())

        # Merge dicts for constructing parameter space
        current_params = {**graph_initial_params, **epidemic_initial_params}
        delta_params = {**graph_delta_params, **epidemic_delta_params}
        delta_fine_params = {**graph_delta_fine_params, **epidemic_delta_fine_params}

        iteration_i = 0

        fine = False  # If False, we are in the first phase. If True, we are in the second phase.
        end = False

        start = time.time()

        # Keep descending the gradient until we reach a local optimum,
        # i.e., the previous set of best parameters is equal to the
        # current set of best parameters.
        while not end:
            iteration_i += 1
            print(f"Iteration {iteration_i} params={current_params}")

            # Construct the search space with the following format.
            #  {"a": [current_params["a"]-delta_params["a"], current_params["a"], current_params["a"]+delta_params["a"],
            #   ...,
            #   "z": [current_params["z"]-delta_params["z"], current_params["z"], current_params["z"]+delta_params["z"]}
            search_space_params = {}
            if not fine:
                for k, v in current_params.items():
                    search_space_params[k] = [v - delta_params[k], v, v + delta_params[k]]
            else:
                for k, v in current_params.items():
                    search_space_params[k] = [v - delta_fine_params[k], v, v + delta_fine_params[k]]

            # Generate the a list of parameters, based on the search space, over
            # which simulations will be run. Our goal is to find the best set of
            # parameters among these.
            grid_params = list(ParameterGrid(search_space_params))

            # Initialize the loss array (RMSE)
            loss = np.full(len(grid_params), np.inf)

            for i, grid_i in enumerate(grid_params):
                # Split grid into epidemic parameters and graph parameters
                graph_grid_i = {**{k: grid_i[k] for k in graph_param_names}, **{"n": self.n_nodes}}
                epidemic_grid_i = {k: grid_i[k] for k in epidemic_param_names}

                print(graph_grid_i, epidemic_grid_i)

                # Skip invalid grids
                if not self.generator_check(**graph_grid_i) \
                        or not Epidemic.parameter_check(n_infected_init=self.n_infected_init, **epidemic_grid_i):
                    continue

                graph = self.generator(**graph_grid_i)

                epidemic = Epidemic('sirv', graph, self.steps,
                                    n_infected_init=self.n_infected_init, vacc=self.vacc, **epidemic_grid_i)

                # Perform simulations_per_grid in order to find a significant
                # result for newly infected nodes per week (variability is too
                # high if we perform a single simulation)
                ni = np.zeros((simulations_per_grid, self.steps+1))

                for sim_id in range(simulations_per_grid):
                    sim = epidemic.simulate()

                    ni[sim_id] = np.array(
                        [self.n_infected_init] +
                        [((sim[t - 1] == 0) & (sim[t] == 1)).sum() for t in range(1, self.steps+1)],
                        dtype=int
                    )

                ni = ni.mean(axis=0)

                # Compute the loss
                loss[i] = self.rmse(ni, self.ni_target)

            prev_params = deepcopy(current_params)
            current_params = grid_params[int(np.argmin(loss))]

            print(f"Lowest loss {np.min(loss)} for grid set {current_params}")

            if self.isclose(current_params, prev_params):
                if not fine:
                    print("Switching to finer delta grid")
                    fine = True
                else:
                    end = True

        print(f"Best parameter set {current_params} after {iteration_i} iteration(s)")
        print(f"Time elapsed: {time.time() - start}")

    @staticmethod
    def rmse(ni, ni_target):
        n_steps = len(ni) - 1
        return np.sqrt((1 / n_steps) * ((ni - ni_target) ** 2).sum())

    @staticmethod
    def isclose(dict1, dict2):
        for k in dict1.keys():
            if not np.isclose(dict1[k], dict2[k]):
                return False
        return True


class ParameterGrid:
    """scikit-learn (0.24.1) -- sklearn.model_selection._search.ParameterGrid"""

    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                            'a list ({!r})'.format(param_grid))

        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        # check if all entries are dictionaries of lists
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError('Parameter grid is not a '
                                'dict ({!r})'.format(grid))
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError('Parameter grid value is not iterable '
                                    '(key={!r}, value={!r})'
                                    .format(key, grid[key]))

        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid."""
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        prod = partial(reduce, operator.mul)
        return sum(prod(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration"""
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')