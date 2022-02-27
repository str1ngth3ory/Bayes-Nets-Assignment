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

import random
import collections
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
    return alarm_prob[1]


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge
    showing hot in the
    power plant system."""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    gauge_prob = marginal_prob['gauge'].values
    return gauge_prob[1]


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
    return temp_prob[1]


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
    if initial_state == None or initial_state == ():
        sample = [0 for _ in range(6)]
        for i in range(0,3):
            sample[i] = random.randint(0,3)
        sample[3] = 0
        sample[4] = random.randint(0,2)
        sample[5] = 2
        return tuple(sample)

    sample = tuple(initial_state)
    vars = ['A','B','C','AvB','BvC','CvA']
    rand_var = random.choice([0,1,2,4])
    if rand_var == 4:
        match_table = bayes_net.get_cpds('BvC').values
        p_var = [match_table[0][sample[1]][sample[2]], \
                match_table[1][sample[1]][sample[2]], \
                match_table[2][sample[1]][sample[2]]]
        sample = list(sample)
        sample[4] = random.choices([0,1,2],weights=p_var)[0]
        # solver = VariableElimination(bayes_net)
        # conditional_prob = solver.query(variables=[vars[4]], \
        #                         evidence={'A':sample[0],'B':sample[1],'C':sample[2],'AvB':sample[3],'CvA':sample[5]}, joint=False)
        # print(p_var)
        # print(conditional_prob[vars[rand_var]].values)
    else:
        p_var = []
        a = vars[rand_var]
        if rand_var == 0:
            b1,b2,c1,c2,a2 = 'B','C','AvB','CvA','BvC'
            b1_idx,b2_idx,c1_idx,c2_idx,a2_idx = 1,2,3,5,4
        elif rand_var == 1:
            b1,b2,c1,c2,a2 = 'A','C','AvB','BvC','CvA'
            b1_idx,b2_idx,c1_idx,c2_idx,a2_idx = 0,2,3,4,5
        elif rand_var == 2:
            b1,b2,c1,c2,a2 = 'A','B','CvA','BvC','AvB'
            b1_idx,b2_idx,c1_idx,c2_idx,a2_idx = 0,1,5,4,3
        mt_c1 = bayes_net.get_cpds(c1).values
        mt_c2 = bayes_net.get_cpds(c2).values
        tt_a = bayes_net.get_cpds(a).values
        tt_b1 = bayes_net.get_cpds(b1).values
        tt_b2 = bayes_net.get_cpds(b2).values
        p_a_c1b1c2b2 = list()
        p_b1 = tt_b1[sample[b1_idx]]
        p_b2 = tt_b2[sample[b2_idx]]
        for i in range(4):
            p_a = tt_a[i]
            if c1[0] == a:
                p_a_v_b1 = mt_c1[sample[c1_idx]][i][sample[b1_idx]]
            else:
                p_a_v_b1 = mt_c1[sample[c1_idx]][sample[b1_idx]][i]
            if c2[0] == a:
                p_a_v_b2 = mt_c2[sample[c2_idx]][i][sample[b2_idx]]
            else:
                p_a_v_b2 = mt_c2[sample[c2_idx]][sample[b2_idx]][i]
            p_a_c1b1c2b2.append(p_a*p_b1*p_b2*p_a_v_b1*p_a_v_b2)
        # print(p_a_c1b1c2b2)
        p_var = [p_a_c1b1c2b2[i]/sum(p_a_c1b1c2b2) for i in range(4)]
        # print(p_var)
        # solver = VariableElimination(bayes_net)
        # conditional_prob = solver.query(variables=[a], \
        #                         evidence={b1:sample[b1_idx],b2:sample[b2_idx],c1:sample[c1_idx],c2:sample[c2_idx],a2:sample[a2_idx]}, joint=False)
        # print(conditional_prob[vars[rand_var]].values)
        # print(conditional_prob[a])
        sample = list(sample)
        sample[rand_var] = random.choices([0,1,2,3],weights=p_var)[0]
    return tuple(sample)


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value.
    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    """
    def _joint_p(state):
        a, b, c, avb, bvc, cva = state
        mt_avb = bayes_net.get_cpds('AvB').values
        mt_bvc = bayes_net.get_cpds('BvC').values
        mt_cva = bayes_net.get_cpds('CvA').values
        tt_a = bayes_net.get_cpds('A').values
        tt_b = bayes_net.get_cpds('B').values
        tt_c = bayes_net.get_cpds('C').values
        return tt_a[a]*tt_b[b]*tt_c[c]*mt_avb[avb][a][b]*mt_bvc[bvc][b][c]*mt_cva[cva][c][a]

    sample = [0 for _ in range(6)]
    for i in range(0,3):
        sample[i] = random.randint(0,3)
    sample[3] = 0
    sample[4] = random.randint(0,2)
    sample[5] = 2

    if initial_state == None or initial_state == ():
        return tuple(sample)

    u = random.random()
    p_x_t = _joint_p(sample)
    p_x_t_1 = _joint_p(initial_state)
    if u <= (p_x_t/p_x_t_1):
        return tuple(sample)
    else:
        return initial_state


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""
    def _ev_sampling(N, sampling_method, samples, initial_state):
        count = 0
        reject_count = 0
        p_var = []
        temp = tuple()
        if sampling_method is Gibbs_sampler:
            while count < N:
                initial_state = sampling_method(bayes_net, initial_state)
                samples.append(initial_state[4])
                count += 1
        elif sampling_method is MH_sampler:
            bool_reject = 0
            while count < N:
                temp = sampling_method(bayes_net, initial_state)
                samples.append(temp[4])
                count += 1
                if temp is initial_state:
                    reject_count += 1
                initial_state = temp
        samples_freq = collections.Counter(samples)
        for i in range(3):
            p_var.append(samples_freq[i]/sum(samples_freq.values()))
        ev = 0
        for i in range(3):
            ev += i*p_var[i]
        if sampling_method is Gibbs_sampler:
            return ev, p_var
        elif sampling_method is MH_sampler:
            return ev, p_var, reject_count

    N = 500
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    if initial_state != None and initial_state != ():
        Gibbs_count += 1
        MH_count += 1

    samples = []
    delta = 1
    last_ev = 0
    current_ev, Gibbs_convergence = _ev_sampling(N, Gibbs_sampler, samples, initial_state)
    Gibbs_count += N
    while delta > 0.0001:
        last_ev = current_ev
        current_ev, Gibbs_convergence = _ev_sampling(N, Gibbs_sampler, samples, initial_state)
        delta = abs(current_ev - last_ev)
        Gibbs_count += N

    samples = []
    delta = 1
    last_ev = 0
    current_ev, MH_convergence, rej_count = _ev_sampling(N, MH_sampler, samples, initial_state)
    MH_count += N
    MH_rejection_count += rej_count
    while delta > 0.001:
        last_ev = current_ev
        current_ev, MH_convergence, rej_count = _ev_sampling(N, MH_sampler, samples, initial_state)
        delta = abs(current_ev - last_ev)
        MH_count += N
        MH_rejection_count += rej_count

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 4
    return options[1], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return 'Zi Sang'
