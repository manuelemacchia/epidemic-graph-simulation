import argparse
import numpy as np
import os.path
from simulation.model.epidemic import Epidemic
from simulation.model.search import ParameterSearch
from simulation.network import generator
from simulation.plot.plot import sir_plot, ni_plot, ninv_plot


def check_strict_positive(value):
    int_value = int(value)
    if int_value < 1:
        raise argparse.ArgumentTypeError(f"{value} is an invalid strictly positive int value")
    return int_value


def check_probability(value):
    float_value = float(value)
    if float_value < 0 or float_value > 1:
        raise argparse.ArgumentTypeError(f"{value} is an invalid probability")
    return float_value


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-s', '--steps',
                        default=10,
                        type=check_strict_positive,
                        help='number of steps (e.g., weeks) of the simulation\n'
                             'note: if a vaccination vector is specified, this is inferred')
    parser.add_argument('-v', '--vaccination',
                        nargs='+',
                        default=None,
                        help='list with total fraction (pct) of population that has received vaccination by each week\n'
                             'note: if specified, the epidemic model is SIRV, otherwise SIR')
    parser.add_argument('-i', '--iterations',
                        default=1,
                        type=check_strict_positive,
                        help='number of independent simulations to perform\n'
                             'if more than 1, the final configuration is an average across all simulations')
    parser.add_argument('-b', '--beta',
                        default=0.3,
                        type=check_probability,
                        help='probability of spreading from an infected to a susceptible during one time step')
    parser.add_argument('-r', '--rho',
                        default=0.6,
                        type=check_probability,
                        help='probability that an infected individual will recover during one time step')
    parser.add_argument('-ii', '--initialinfect',
                        default=10,
                        type=check_strict_positive,
                        help='number of infected nodes in the initial configuration, chosen uniformly among all nodes')
    parser.add_argument('-g', '--graph',
                        default='pa',
                        choices=['kr', 'pa', 'nws'],
                        help='random graph model:\n'
                             '`kr`: k-regular graph\n'
                             '`pa`: preferential attachment\n'
                             '`nws`: Newman-Watts-Strogatz small world')
    parser.add_argument('-n', '--nodes',
                        default=500,
                        type=check_strict_positive,
                        help='total number of nodes of the graph')
    parser.add_argument('-k',
                        default=4,
                        type=check_strict_positive,
                        help='model `kr`: neighbors of each node (k-1 if k odd)\n'
                             'model `pa`: average degree of the final graph\n'
                             'model `nws`: neighbors of each node in the underlying graph (k-1 if k odd)')
    parser.add_argument('-p',
                        default=0.5,
                        type=check_probability,
                        help='probability of creating a new edge\n'
                             'note: only valid for model `nws`')
    parser.add_argument('--sweden',
                        dest='simulate_sweden',
                        action='store_true',
                        help='estimate the parameters of the H1N1 pandemic in Sweden 2009\n'
                             'note: beta, rho, k will be taken as starting point for gradient descent')
    parser.add_argument('-o', '--output',
                        required=True,
                        help='path to save output of the simulation (plots and configuration)')

    parser_args = parser.parse_args()
    return parser_args


def simulate(iterations, graph_generator, graph_params, n_nodes, beta, rho, steps, n_infected_init, vacc=None):
    """Perform `iterations` simulations and compute averages. If vacc is not
    None, run the simulation using the SIRV model, otherwise use SIR."""

    # Initialize arrays for computing averages over simulations
    s = np.zeros((iterations, steps + 1), dtype=int)
    i = np.zeros((iterations, steps + 1), dtype=int)
    r = np.zeros((iterations, steps + 1), dtype=int)
    ni = np.zeros((iterations, steps + 1), dtype=int)
    if vacc is not None:
        v = np.zeros((iterations, steps + 1), dtype=int)
        nv = np.zeros((iterations, steps + 1), dtype=int)

    for sim_id in range(iterations):
        graph = graph_generator(**{'n': n_nodes, **graph_params})

        if vacc is not None:
            epidemic = Epidemic('sirv', graph, steps,
                                beta=beta, rho=rho, n_infected_init=n_infected_init, vacc=vacc)
        else:
            epidemic = Epidemic('sir', graph, steps,
                                beta=beta, rho=rho, n_infected_init=n_infected_init)

        sim = epidemic.simulate()

        # Compute four (steps, ) array containing the total number, at each
        # step, of susceptible (S), infected (I), recovered (R) and vaccinated
        # (V) respectively.
        s[sim_id] = np.ma.masked_not_equal(sim, 0).count(axis=1)
        i[sim_id] = np.ma.masked_not_equal(sim, 1).count(axis=1)
        r[sim_id] = np.ma.masked_not_equal(sim, 2).count(axis=1)
        if vacc is not None:
            v[sim_id] = np.ma.masked_not_equal(sim, 3).count(axis=1)

        # Compute a (steps, ) array containing the number of newly infected
        # individuals at each step. The number of newly infected at time t is
        # defined as the sum of nodes that went from state 0 (S) at time t-1
        # to state 1 (I) at time t.
        ni[sim_id] = np.array(
            [n_infected_init] + [((sim[t - 1] == 0) & (sim[t] == 1)).sum() for t in range(1, steps + 1)],
            dtype=int)

        # Compute the same kind of array for newly vaccinated individuals.
        if vacc is not None:
            nv[sim_id] = np.array(
                [v[sim_id, 0]] + [((sim[t - 1] != 3) & (sim[t] == 3)).sum() for t in range(1, steps + 1)],
                dtype=int)

    # Compute the average total number of susceptible, infected, recovered and
    # vaccinated nodes at each week.
    s = s.mean(axis=0)
    i = i.mean(axis=0)
    r = r.mean(axis=0)
    if vacc is not None:
        v = v.mean(axis=0)

    # Compute the average number of newly infected and vaccinated individuals
    # each week.
    ni = ni.mean(axis=0)
    if vacc is not None:
        nv = nv.mean(axis=0)

    if vacc is not None:
        return s, i, r, v, ni, nv
    else:
        return s, i, r, ni


def sweden_parameter_search(basic=False, initial_params=None):
    ps = ParameterSearch(generator.preferential_attachment,
                         generator.preferential_attachment_check,
                         n_nodes=934,
                         steps=15,
                         n_infected_init=1,
                         vacc=[5, 9, 16, 24, 32, 40, 47, 54, 59, 60, 60, 60, 60, 60, 60],
                         ni_target=[1, 1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0])

    if initial_params is None:  # Set default parameters
        initial_params = {"k": 10, "beta": 0.3, "rho": 0.6}

    graph_initial_params = {"k": initial_params["k"]}
    epidemic_initial_params = {"beta": initial_params["beta"], "rho": initial_params["rho"]}

    graph_delta_params = {"k": 1}
    epidemic_delta_params = {"beta": 0.1, "rho": 0.1}

    if basic:
        ps.basic_search(graph_initial_params,
                        epidemic_initial_params,
                        graph_delta_params,
                        epidemic_delta_params,
                        simulations_per_grid=10)
    else:
        graph_delta_fine_params = {"k": 1}
        epidemic_delta_fine_params = {"beta": 0.025, "rho": 0.025}

        ps.search(graph_initial_params,
                  epidemic_initial_params,
                  graph_delta_params,
                  epidemic_delta_params,
                  graph_delta_fine_params,
                  epidemic_delta_fine_params,
                  simulations_per_grid=10)

    return ps.best_params


def save_output(averages, args):
    # Save outputs (configuration and plots) to disk
    if args.vaccination is not None:
        s, i, r, v, ni, nv = averages
        sir_plot(s, i, r, v, file=os.path.join(args.output, "sirv.png"))
        ninv_plot(ni, nv, file=os.path.join(args.output, "ninv.png"))
    else:
        s, i, r, ni = averages
        sir_plot(s, i, r, file=os.path.join(args.output, "sir.png"))
        ni_plot(ni, file=os.path.join(args.output, "ni.png"))

    with open(os.path.join(args.output, "avgs.txt"), 'w') as f:
        f.write(f"s\t{s}\n"
                f"i\t{i}\n"
                f"r\t{r}\n"
                f"ni\t{ni}\n")
        if args.vaccination is not None:
            f.write(f"v\t{ni}\n"
                    f"nv\t{nv}\n")

    with open(os.path.join(args.output, "params.txt"), 'w') as f:
        f.write(f"{args}")


if __name__ == '__main__':
    args = parse_args()

    if args.simulate_sweden:
        initial_params = {
            "k": args.k,
            "beta": args.beta,
            "rho": args.rho
        }

        print("Starting parameter search for the Sweden 2009 pandemic")
        best_params = sweden_parameter_search(initial_params=initial_params)

    else:
        generators = {
            'kr': 'k_regular',
            'pa': 'preferential_attachment',
            'nws': 'newman_watts_strogatz'
        }

        graph_generator = getattr(generator, generators[args.graph])
        graph_params = {'k': args.k}
        if args.graph == 'nws':
            graph_params = {**graph_params, 'p': args.p}

        print("Starting simulation with parameters:\n"
              f"    number of simulations = {args.iterations}\n"
              f"    number of steps of the simulation = {args.steps}\n"
              f"    graph family = {generators[args.graph]}\n"
              f"    graph params = {graph_params}\n"
              f"    number of nodes = {args.nodes}\n"
              f"    beta = {args.beta}\n"
              f"    rho = {args.rho}\n"
              f"    number of initially infected = {args.initialinfect}")
        if args.vaccination is not None:
            args.vaccination = list(map(int, args.vaccination))
            print(f"    epidemic model = SIRV\n"
                  f"    vaccination vector = {args.vaccination}")
        else:
            print(f"    epidemic model = SIR")

        # Perform simulation and retrieve averages over #iterations
        averages = simulate(iterations=args.iterations,
                            graph_generator=graph_generator,
                            graph_params=graph_params,
                            n_nodes=args.nodes,
                            beta=args.beta,
                            rho=args.rho,
                            steps=args.steps,
                            n_infected_init=args.initialinfect,
                            vacc=args.vaccination)

        save_output(averages, args)
