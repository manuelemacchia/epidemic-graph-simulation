import argparse
from simulation.network.generator import Graph
from simulation.model.epidemic import Epidemic
from simulation.model.search import ParameterSearch


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument(...)

    args = parser.parse_args()
    return args


def main(args):
    print("Hello world!")


def simulate(args): # TODO this should basically be multiple_simulations
    print("simulate")


if __name__ == '__main__':
    args = parse_args()
    main(args)

    # DEBUG
    graph = Graph('k_regular', k=6, n=100)
    graph = Graph('preferential_attachment', k=5, n=100)
