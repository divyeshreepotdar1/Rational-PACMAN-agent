# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

BIGNUM = 99999999

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            states = self.mdp.getStates()
            Counter  = util.Counter()
            for state in states:
                max_val = -BIGNUM
                for action in self.mdp.getPossibleActions(state):
                    Qvalue = self.computeQValueFromValues(state, action)
                    if Qvalue > max_val:
                        max_val = Qvalue
                    Counter [state] = max_val
            self.values = Counter


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        T_of_SA = self.mdp.getTransitionStatesAndProbs(state, action)
        V_of_S = 0
        for s_prime, prob in T_of_SA:
            reward = self.mdp.getReward(state, action, s_prime)
            V_of_S += prob * (reward + self.discount * self.values[s_prime])
        return V_of_S
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        Ideal_Act = None
        Max_S_Val = -BIGNUM
        for action in self.mdp.getPossibleActions(state):
          Q_of_SA = self.computeQValueFromValues(state, action)
          if Q_of_SA > Max_S_Val:
            Max_S_Val = Q_of_SA
            Ideal_Act = action
        return Ideal_Act
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            state = self.mdp.getStates()[i %  len(self.mdp.getStates())]
            val = self.computeActionFromValues(state)
            if val is None:
                self.values[state] = 0
            else:
                self.values[state] = self.computeQValueFromValues(state, val)
            

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
    
    def computeQValues(self, state):
        # Returns a counter containing all qValues from a given state

        actions = self.mdp.getPossibleActions(state)  # All possible actions from a state
        qValues = util.Counter()  # A counter holding (action, qValue) pairs

        for action in actions:
            # Putting the calculated Q value for the given action into my counter
            qValues[action] = self.computeQValueFromValues(state, action)

        return qValues

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        World_States = self.mdp.getStates()
        S_nbr = dict()
        for state in World_States:
            S_nbr[state]=set()
        for state in World_States:
            Val_Acts = self.mdp.getPossibleActions(state)
            for a in Val_Acts:
                S_Primes = self.mdp.getTransitionStatesAndProbs(state, a)
                for nextState,pred in S_Primes:
                    if pred>0:
                        S_nbr[nextState].add(state)
        
        pq = util.PriorityQueue()
        for state in World_States:

            Q_of_S = self.computeQValues(state)

            if len(Q_of_S) > 0:
                max_of_QS = Q_of_S[Q_of_S.argMax()]
                diff = abs(self.values[state] - max_of_QS)
                pq.push(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                return;
            state = pq.pop()
            Q_of_S = self.computeQValues(state)
            max_of_QS = Q_of_S[Q_of_S.argMax()]
            self.values[state] = max_of_QS
            for p in S_nbr[state]:

                S_nbr_Qval = self.computeQValues(p)
                max_of_QS = S_nbr_Qval[S_nbr_Qval.argMax()]
                diff = abs(self.values[p] - max_of_QS)

                if diff > self.theta:
                    pq.update(p, -diff)

