import numpy as np


class Epidemic:
    """Simulate an epidemic in discrete time according to the SIR and SIRV models"""

    def __init__(self, model, graph, steps, **kwargs):
        """Set the epidemic model, the number of steps of the simulation and
         create a graph of the specified type on which to simulate the epidemic.

        Args:
            model: either 'sir' or 'sirv'
            graph: graph representing individuals (nodes) and connections
                between these individuals (edges)
            steps: number of steps, e.g., weeks, of the simulation
            **kwargs: parameters for the epidemic model

        """

        self.graph = graph
        self.steps = steps

        self.model = model
        self.simulation_args = kwargs

    def simulate(self):
        return getattr(self, self.model)(**self.simulation_args)

    def sir(self, beta, rho, n_infected_init):
        """Simulate an epidemic in discrete time according to a simplified
        version of the SIR model.

        Args:
            beta: probability that the infection is spread from an infected
                individual to a susceptible one, given that they are connected
                by a link, during one time step
            rho: probability that an infected individual will recover during one
                time step
            n_infected_init: number of infected nodes in the initial
                configuration, chosen randomly among all nodes of the graph
                according to a uniform probability distribution

        Returns:
            config ((n_steps+1, n_nodes) numpy array): configuration of nodes at
                every step of the simulation

        """

        n_nodes = len(self.graph.nodes)

        # (n_steps, n_nodes) array containing the configuration of the
        # nodes at each step of the simulation. Note that we consider
        # steps from t=0 (initial configuration) to t=n_steps included,
        # so we store n_steps+1 steps in total.
        config = np.zeros((self.steps + 1, n_nodes), dtype=int)

        # Randomly choose n_infected_init infected nodes at the beginning
        # of the simulation (t=0)
        config[0, np.random.choice(range(n_nodes), n_infected_init, replace=False)] = 1

        for t in range(1, self.steps+1):
            for i in range(n_nodes):
                # Get the last state of the current node at time t-1
                state = config[t-1, i]

                if state == 0:  # State of the node is susceptible (S)
                    # Get the state of neighbors of node i at time t-1
                    neighbor_ids = [e[1] for e in self.graph.edges(i)]
                    neighbor_states = config[t-1, neighbor_ids]

                    # Compute the number of infected neighbors of i at time t-1
                    m = np.ma.masked_not_equal(neighbor_states, 1).filled(0).sum()

                    # Compute transition probabilities for node i
                    #           S (0)         I (1)
                    prob = [(1 - beta) ** m, 1 - (1 - beta) ** m]

                    # Update the configuration at time t
                    config[t, i] = np.random.choice([0, 1], p=prob)

                elif state == 1:  # State of the node is infected (I)
                    #       I (1)  R (2)
                    prob = [1 - rho, rho]

                    config[t, i] = np.random.choice([1, 2], p=prob)

                else:  # State of the node is recovered (R)
                    config[t, i] = 2

        return config

    def sirv(self, beta, rho, n_infected_init, vacc):
        """Simulate an epidemic in discrete time according to a modified version
        of the SIR model which accounts for vaccinations.

        Args:
            beta: probability that the infection is spread from an infected
                individual to a susceptible one, given that they are connected
                by a link, during one time step
            rho: probability that an infected individual will recover during one
                time step
            n_infected_init: number of infected nodes in the initial
                configuration, chosen randomly among all nodes of the graph
                according to a uniform probability distribution
            vacc: total fraction of population that has received vaccination by
                each week

        Returns:
            config ((n_steps+1, n_nodes) numpy array): configuration of nodes at
                every step of the simulation

        """

        n_nodes = len(self.graph.nodes)

        config = np.zeros((self.steps + 1, n_nodes), dtype=int)

        # Randomly choose n_infected_init infected nodes at the beginning
        # of the simulation (t=0)
        config[0, np.random.choice(range(n_nodes), n_infected_init, replace=False)] = 1

        # In the last week t=n_steps, we should take into account the
        # cumulative vaccination percent of week n_steps+1, which does
        # not appear in the vacc array. Therefore, we assume that
        # vaccinations remain stable.
        vacc.append(vacc[-1])

        # Start the simulation and keep iterating until t=15 (included)
        # We consider the time between one step and another to be a week,
        # i.e., (t-1, t] is week t. Vaccinations of week t are given
        # at the beginning of the week, so at time (t-1)^+, and are
        # counted at the step t on the plots.
        for t in range(1, self.steps + 1):
            # The vaccination is assumed to take effect immediately once
            # given, i.e., if a node is vaccinated in week t, it can not
            # infect any other node during that week.
            #
            # Therefore, firstly we select individuals to vaccinate.
            # These individuals are selected uniformly at random from the
            # population that has not yet received vaccination, regardless
            # of their state (either S, I or R).

            # Calculate the percent of the population that should receive
            # a vaccination this week.
            pct_vacc = vacc[(t - 1) + 1] - vacc[(t - 1)]

            # Calculate how many nodes should receive a vaccination this week
            # (equal to pct_vacc% of the total number of nodes)
            n_vacc = int(np.floor(n_nodes * pct_vacc / 100))

            # Randomly choose n_vacc nodes to vaccinate. All nodes are
            # uniquely chosen (no individual may receive more than 1 vaccine)
            idx_unvacc = np.argwhere(config[t - 1] != 3).flatten()  # Indices of all not-yet-vaccinated nodes
            idx_vacc = np.random.choice(idx_unvacc, n_vacc, replace=False)  # Indices of nodes chosen to vaccinate

            # Iterate over each node
            for i in range(n_nodes):
                # Get the state of the current node at time t-1.
                # Take into account vaccinated nodes, and set the state
                # to 3 if the current node i has been vaccinated at the
                # beginning of the current week.
                state = 3 if i in idx_vacc else config[t - 1, i]

                if state == 0:  # State of the node is susceptible (S)
                    # Get the state of neighbors of node i at time t-1
                    neighbor_ids = [e[1] for e in self.graph.edges(i)]
                    neighbor_states = config[t - 1, neighbor_ids]

                    # Compute the number of infected neighbors of i at time t-1
                    m = np.ma.masked_not_equal(neighbor_states, 1).filled(0).sum()

                    # Compute transition probabilities for node i.
                    # prob[j] corresponds to the probability for node i of
                    # transitioning from its state (S) at time t-1 to state j
                    # at time t, based on m.
                    #           (S)           (I)
                    prob = [(1 - beta) ** m, 1 - (1 - beta) ** m]

                    # Select the next state of node i according to the
                    # probability distribution we computed previously.
                    config[t, i] = np.random.choice([0, 1], p=prob)

                elif state == 1:  # State of the node is infected (I)
                    #        (I)   (R)
                    prob = [1 - rho, rho]
                    config[t, i] = np.random.choice([1, 2], p=prob)

                elif state == 2:  # State of the node is recovered (R)
                    config[t, i] = 2

                else:  # State of the node is vaccinated (V)
                    # Both nodes that were previously vaccinated and
                    # nodes who are vaccinated at the beginning of
                    # the current week are set to state 3 (V)
                    config[t, i] = 3

        return config

    @staticmethod
    def parameter_check(beta, rho, n_infected_init):
        """Check validity of epidemic parameters. Return True if valid."""
        if (beta < 0 or beta > 1) or (rho < 0 or rho > 1) or (n_infected_init <= 0):
            return False
        return True
