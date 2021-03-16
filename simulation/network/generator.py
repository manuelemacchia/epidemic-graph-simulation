import numpy as np
import networkx as nx


def k_regular(n, k):
    """Generate a symmetric k-regular graph with n nodes.

    Args:
        n: number of nodes
        k: neighbors of each node (even, k/2 on the left and k/2 on the right),
            replaced by k-1 if k odd

    Returns:
        graph: instance of a k-regular graph

    """

    if not k_regular_check(n, k):
        raise ValueError("Invalid parameters")

    nodes = list(range(n))
    edges = []

    if k % 2 != 0:
        k = k-1

    # Generate relative neighbor ids. For instance, with k=4:
    #   [-2, -1, 1, 2]
    relative_neighbors = np.concatenate([np.arange(-k/2, 0), np.arange(1, k/2+1)])

    for node in nodes:
        for neighbor in relative_neighbors:
            # Add an edge between each node and the four nodes whose index
            # is closest to the node's own index
            edges.append((node, nodes[(node + int(neighbor)) % n]))

    graph = nx.Graph()
    graph.add_nodes_from(nodes)  # Needed to keep nodes in ascending order
    graph.add_edges_from(edges)

    return graph


def k_regular_check(n, k):
    if (k > n) or (n <= 0) or (k <= 1):  # n and k must be strictly positive integers
        return False

    return True


def preferential_attachment(n, k):
    """Generate a random graph according to the preferential attachment model.

    At time t=1, start with an initial graph G_1 that is complete with k+1
    nodes in total. Then, at each subsequent time step t, we create a new
    graph G_t by adding a new node and connecting it to some of the existing
    nodes, according to the preferential attachment rule.

    Args:
        n: total number of nodes of the final graph
        k: average degree of the final graph

    Returns:
        graph: instance of a preferential attachment graph

    """

    if not preferential_attachment_check(n, k):
        raise ValueError("Invalid parameters")

    graph = nx.complete_graph(k+1)

    for n_t in range(len(graph.nodes), n):
        # Get out degree of nodes of G_{t-1}
        w = np.array(list(dict(graph.degree()).values()))

        # Compute probabilities of creating an edge between n_t and a node i
        # belonging to G_{t-1} according to the preferential attachment rule
        prob = w/w.sum()

        # Compute the degree that the newly added node will have.
        # Note that, in case k is odd, then k/2 will not be an integer, thus
        # we alternate ceil(k/2) and floor(k/2) to achieve over a large
        # number of nodes an average degree of k/2. This logic has no effect
        # when k is even.
        w_n_t = int(np.ceil(k/2) if n_t % 2 == 0 else np.floor(k/2))

        # Create w_n_t new (undirected) edges to node n_t, according to the
        # probabilities we previously computed. We do not add multiple links
        # to the same node (take w_n_t samples from [0, n_t) without
        # replacement, according to prob).
        idx = np.random.choice(n_t, w_n_t, replace=False, p=prob)
        edges = [(n_t, i) for i in idx]

        # Update the graph from G_{t-1} to G_t
        graph.add_edges_from(edges)

        n_t += 1

    return graph


def preferential_attachment_check(n, k):
    # n and k must be strictly positive integers
    if (k > n) or (n <= 0) or (k <= 1):
        return False

    return True


def newman_watts_strogatz(n, k, p):
    """Generate a random graph according to the Newman-Watts-Strogatz model.

    Firstly, it creates a ring G_u over the n nodes, each node connected to
    k neighbors, k/2 on each side (or k-1 if k is odd). For each edge (i, j)
    in G_u it adds an edge (i, w) with a randomly chosen existing node w
    with probability p.

    Args:
        n: total number of nodes of the graph
        k: each node is joined with its k (or k-1 if k odd) nearest neighbors
        p: probability of adding a new edge for each edge

    Returns:
        graph: instance of a Newman-Watts-Strogatz graph

    """

    return nx.newman_watts_strogatz_graph(n, k, p)


def newman_watts_strogatz_check(n, k, p):
    # n and k must be strictly positive integers
    if (k > n) or (n <= 0) or (k <= 1) or (p < 0 or p > 1):
        return False

    return True
