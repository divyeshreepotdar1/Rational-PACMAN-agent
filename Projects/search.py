# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def recdfs(Prblm, S, CSet): 
         if S.isEmpty():
             return 
         currentState, actions, currentCost = S.pop()
         if Prblm.isGoalState(currentState):
            return actions
         else: 
            if currentState not in CSet:
              CSet.append(currentState)
              #list of (successor, action, stepCost)
              successors = Prblm.getSuccessors(currentState)
              for succState, succAction, succCost in successors:
                  newAction = actions + [succAction]
                  newCost = currentCost + succCost
                  newNode = (succState, newAction, newCost)
                  S.push(newNode)
            Agactions = recdfs(Prblm, S, CSet)
            return Agactions

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    import sys
    sys.setrecursionlimit(10000)
    FringeS = util.Stack()
    
    #previously expanded states (for cycle checking), holds states
    ClosedSet = []
    
    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)
    
    FringeS.push(startNode)
    ##ClosedSet.append(startState) 
    return recdfs(problem, FringeS, ClosedSet)

def recbfs(Prblm, Q, CSet): 
         if Q.isEmpty():
             return 
         currentState, actions, currentCost = Q.pop()
         if Prblm.isGoalState(currentState):
            return actions
         else: 
            if currentState not in CSet:
              CSet.append(currentState)
              #list of (successor, action, stepCost)
              successors = Prblm.getSuccessors(currentState)
              for succState, succAction, succCost in successors:
                  newAction = actions + [succAction]
                  newCost = currentCost + succCost
                  newNode = (succState, newAction, newCost)
                  Q.push(newNode)
            Agactions = recbfs(Prblm, Q, CSet)
            return Agactions


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    import sys
    sys.setrecursionlimit(10**6)
    FringeQ = util.Queue()
    
    #previously expanded states (for cycle checking), holds states
    ClosedSet = []
    
    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)
    
    FringeQ.push(startNode)
    ##ClosedSet.append(startState) 
    return recbfs(problem, FringeQ, ClosedSet)
    

def recUcs(Prblm, PQ, CSet): 
         if PQ.isEmpty():
             return 
         currentState, actions, currentCost = PQ.pop()
         if Prblm.isGoalState(currentState):
            return actions
         else: 
            if (currentState not in CSet) or (currentCost < CSet[currentState]):
              CSet[currentState] = currentCost
              #list of (successor, action, stepCost)
              successors = Prblm.getSuccessors(currentState)
              for succState, succAction, succCost in successors:
                  newAction = actions + [succAction]
                  newCost = currentCost + succCost
                  newNode = (succState, newAction, newCost)
                  PQ.update(newNode, newCost)
            Agactions = recUcs(Prblm, PQ, CSet)
            return Agactions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    import sys
    sys.setrecursionlimit(10000)
    FringePQ = util.PriorityQueue()
    
    #previously expanded states (for cycle checking), holds states
    ClosedDict = {}
    
    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)
    
    FringePQ.push(startNode, 0)
    ##ClosedSet.append(startState) 
    return recUcs(problem, FringePQ, ClosedDict)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()

    exploredNodes = [] #holds (state, cost)

    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)

    frontier.push(startNode, 0)

    while not frontier.isEmpty():

        #begin exploring first (lowest-combined (cost+heuristic) ) node on frontier
        currentState, actions, currentCost = frontier.pop()

        #put popped node into explored list
        currentNode = (currentState, currentCost)
        exploredNodes.append((currentState, currentCost))

        if problem.isGoalState(currentState):
            return actions

        else:
            #list of (successor, action, stepCost)
            successors = problem.getSuccessors(currentState)

            #examine each successor
            for succState, succAction, succCost in successors:
                newAction = actions + [succAction]
                newCost = problem.getCostOfActions(newAction)
                newNode = (succState, newAction, newCost)

                #check if this successor has been explored
                already_explored = False
                for explored in exploredNodes:
                    #examine each explored node tuple
                    exploredState, exploredCost = explored

                    if (succState == exploredState) and (newCost >= exploredCost):
                        already_explored = True

                #if this successor not explored, put on frontier and explored list
                if not already_explored:
                    frontier.push(newNode, newCost + heuristic(succState, problem))
                    exploredNodes.append((succState, newCost))

    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
