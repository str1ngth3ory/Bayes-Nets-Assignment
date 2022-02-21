import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    BayesNet.add_node('temperature')
    BayesNet.add_node('gauge')
    BayesNet.add_node('faulty gauge')
    BayesNet.add_node('alarm')
    BayesNet.add_node('faulty alarm')
    BayesNet.add_edge("temperature","faulty gauge")
    BayesNet.add_edge("temperature","gauge")
    BayesNet.add_edge("faulty gauge","gauge")
    BayesNet.add_edge("faulty alarm","alarm")
    BayesNet.add_edge("gauge","alarm")
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    cpd_t = TabularCPD('temperature', 2, [[0.8],[0.2]])
    cpd_fa = TabularCPD('faulty alarm', 2, [[0.85],[0.15]])
    cpd_fg = TabularCPD('faulty gauge', 2, [[0.95,0.2], \
            [0.05,0.8]], evidence=['temperature'], evidence_card=[2])
    cpd_g = TabularCPD('gauge', 2, [[0.95,0.05,0.2,0.8], \
            [0.05,0.95,0.8,0.2]], evidence=['faulty gauge','temperature'], evidence_card=[2,2])
    cpd_a = TabularCPD('alarm', 2, [[0.9,0.1,0.55,0.45], \
            [0.1,0.9,0.45,0.55]], evidence=['faulty alarm','gauge'], evidence_card=[2,2])
    bayes_net.add_cpds(cpd_t, cpd_fa, cpd_fg, cpd_g, cpd_a)
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal
    probability of the alarm
    ringing in the
    power plant system."""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    alarm_prob = marginal_prob['alarm'].values
    return alarm_prob


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge
    showing hot in the
    power plant system."""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    gauge_prob = marginal_prob['gauge'].values
    return gauge_prob


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'], \
                        evidence={'alarm':1,'faulty alarm':0, 'faulty gauge':0}, joint=False)
    temp_prob = conditional_prob['temperature'].values
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    BayesNet.add_node('A')
    BayesNet.add_node('B')
    BayesNet.add_node('C')
    BayesNet.add_node('AvB')
    BayesNet.add_node('BvC')
    BayesNet.add_node('CvA')
    BayesNet.add_edge("A","AvB")
    BayesNet.add_edge("B","AvB")
    BayesNet.add_edge("B","BvC")
    BayesNet.add_edge("C","BvC")
    BayesNet.add_edge("C","CvA")
    BayesNet.add_edge("A","CvA")
    cpd_a = TabularCPD('A', 4, [[0.15],[0.45],[0.3],[0.1]])
    cpd_b = TabularCPD('B', 4, [[0.15],[0.45],[0.3],[0.1]])
    cpd_c = TabularCPD('C', 4, [[0.15],[0.45],[0.3],[0.1]])
    cpd_ab = TabularCPD('AvB', 3, \
            [[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1], \
            [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1], \
            [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]], \
            evidence=['A','B'], evidence_card=[4,4])
    cpd_bc = TabularCPD('BvC', 3, \
            [[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1], \
            [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1], \
            [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]], \
            evidence=['B','C'], evidence_card=[4,4])
    cpd_ca = TabularCPD('CvA', 3, \
            [[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1], \
            [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1], \
            [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]], \
            evidence=['C','A'], evidence_card=[4,4])
    BayesNet.add_cpds(cpd_a, cpd_b, cpd_c, cpd_ab, cpd_bc,cpd_ca)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C.
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'], \
                        evidence={'AvB':0,'CvA':2}, joint=False)
    posterior = conditional_prob['BvC'].values
    return posterior # list


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm
    given a Bayesian network and an initial state value.

    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)

    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.
    """
    sample = tuple(initial_state)
    # TODO: finish this function
    raise NotImplementedError
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value.
    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    """
    A_cpd = bayes_net.get_cpds("A")
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = tuple(initial_state)
    # TODO: finish this function
    raise NotImplementedError
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    raise NotImplementedError
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    raise NotImplementedError
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    raise NotImplementedError
