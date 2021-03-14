# Epidemic simulation on graph

## Overview
The goal is to learn the network characteristics and disease dynamics of the pandemic occurred in Sweden during 2009, commonly known as [swine flu](https://en.wikipedia.org/wiki/2009_swine_flu_pandemic). As a secondary goal, we develop an algorithm to simulate an epidemic on a random graph of choice according to the SIR and SIRV models.

→ [Open the notebook](https://nbviewer.jupyter.org/github/manuelemacchia/epidemic-graph-simulation/blob/main/experiments.ipynb) to read the analysis.

## Requirements
All required packages are listed in `requirements.txt`. All experiments in the notebook are carried out on Python 3.9.2.

## Usage
### Simulate SIR/SIRV on a graph
To simulate an epidemic on a (random) graph of choice with custom parameters, run `python simulate.py` with the following arguments.

- `-s/--steps <STEPS>` number of steps to simulate, e.g., weeks.
- `-v/--vaccination <VACCINATION VECTOR>` list with total fraction (in percent) of population that has received vaccination by each week. If specified, the epidemic model will be SIRV, otherwise SIR.
- `-i/--iterations <ITERATIONS>` number of identical independent simulations to run. If more than 1, the output will be an average across simulations.
- `-b/--beta <PROBABILITY: BETA>` probability that an infected individual spreads the infection to a susceptible individual during one time step, e.g., one week.
- `-r/--rho <PROBABILITY: RHO>` probability that an infected individual will recover during one time step, e.g., one week.
- `-ii/--initialinfect <INITIALLY INFECTED>` number of infected nodes in the initial configuration. The specific nodes are chosen at random among all nodes in the population according to a uniform probability distribution.
- `-g/--graph <GRAPH FAMILY>` random graph model, either `kr` (k-regular), `pa` (preferential attachment) or `nws` (Newman-Watts-Strogatz small world).
- `-n/--nodes <NODES>` total number of nodes in the population.
- `-k <K>` parameter k of the graph. The meaning depends on the chosen graph family.
    - For k-regular graphs, neighbors of each node (k-1 if k odd).
    - For preferential attachment graphs, average degree of the final graph.
    - For Newman-Watts-Strogatz small world graphs, neighbors of each node in the underlying graph (k-1 if k odd).
- `-p <P>` probability of creating a new edge (i, w) between each node i such that (i, j) is an edge. Only valid for Newman-Watts-Strogatz small world graphs.
- `--sweden` estimate the parameters of the H1N1 pandemic in Sweden 2009 on a preferential attachment graph. Parameters beta, rho, k will be taken as starting point for the gradient descent. Other parameters are ignored.
- `-o/--output <PATH>` path to save output of the simulation (plots and configuration).

The simulation outputs four files.
- `avgs.txt` contains (average) vectors of the total number of susceptible, infected, recovered and, eventually, vaccinated individuals each week, as well as the (average) number of newly infected individuals each week, and the number of newly vaccinated individuals each week
- `output.txt` contains the parameters specified to run the simulation
- `sir.png` or `sirv.png` is a plot of the average total number of susceptible, infected, recovered and, eventually, vaccinated individuals each week
- `ni.png` or `ninv.png` is a plot of the average number of newly infected individuals each week and, eventually, the number of newly vaccinated individuals each week

Note that the output path specified with `--output` must exist. Any existing file with the same name of the output files will be overwritten.

#### Example
Simulate an epidemic for 10 weeks with the SIRV model, with a vaccination vector `[5, 9, 16, 24, 32, 40, 47, 54, 59, 60]`. Perform 10 simulations in total and retrieve the average number of susceptible, infected and recovered across these simulations.

```
python simulate.py -s 10 -v 5 9 16 24 32 40 47 54 59 60 -i 10 -b 0.4 -r 0.8 -ii 15 -g nws -n 200 -k 6 -p 0.5 -o output
```

### Simulate the Sweden 2009 pandemic
To perform a parameter search with the goal of finding the parameter set that best fits the swine flu pandemic of 2009 in Sweden, run `python simulate.py --sweden`. The parameter search will run with default initial parameters `-k 10 -b 0.3 -r 0.6` unless specified otherwise. The program will display the best set of parameters k, beta and rho found.

## Notes
This project has been assigned as a problem in the Network Dynamics and Learning course at the Polytechnic University of Turin, during A.Y. 2020/2021.

All numbers regarding the H1N1 pandemic in Sweden during the fall of 2009 have been taken from the report by the Swedish Civil Contingencies Agency (*Myndigheten för samhällsskydd och beredskap*, MSB) and the Swedish Institute for Communicable Disease Control (*Smittsky-ddsinstitutet*, SMI).