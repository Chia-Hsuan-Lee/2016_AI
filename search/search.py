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
import sys

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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    #print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())
    #Start: (5, 5)
    #Is the start a goal? False
    #Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]

    frontier = util.Stack() # LIFO  (x-y_coodinate, path to this coordinate )
    explored_set = util.Stack() # FIFO (x-y_coordinate)


    # Is the start a goal?
    if problem.isGoalState(problem.getStartState()):
        print "The start node is the goal, program terminating.... "
	return []
    # initial frontier with start state
    frontier.push(( problem.getStartState(),[]))
    while True:
        # check if the frontier is empty
        if frontier.isEmpty():
            print "The frontier is empty, program terminating.... "
            return []
       
        # choose one node to expand
	half_explore = frontier.pop()
        state = half_explore[0]
        path = half_explore[1]

        # check if the selected_node is the goal
        if problem.isGoalState(state):
            print "goal !!!!"
            print path
	    return path

        explored_set.push(state)

        for successor in problem.getSuccessors(state):
            temp_state = successor[0]
            temp_move = successor[1]

            frontier_state_list = []   
            for node in frontier.list:
                frontier_state_list.append(node[0])
           
	    if temp_state not in frontier_state_list:
                if temp_state not in explored_set.list:
                    temp_path = path + [temp_move]
	            frontier.push(( temp_state, temp_path ))
                   
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue() # LIFO  (x-y_coodinate, path to this coordinate )
    explored_set = util.Stack() # FIFO (x-y_coordinate)


    # Is the start a goal?
    if problem.isGoalState(problem.getStartState()):
        print "The start node is the goal, program terminating.... "
	return []
    # initial frontier with start state
    frontier.push(( problem.getStartState(),[] ))
    while True:
        # check if the frontier is empty
        if frontier.isEmpty():
            print "The frontier is empty, program terminating.... "
            return []
       
        # choose one node to expand
	half_explore = frontier.pop()
        state = half_explore[0]
        path = half_explore[1]

        # check if the selected_node is the goal
        if problem.isGoalState(state):
	    return path

        explored_set.push(state)

        for successor in problem.getSuccessors(state):
            temp_state = successor[0]
            temp_move = successor[1]

            frontier_state_list = []   
            for node in frontier.list:
                frontier_state_list.append(node[0])
           
	    if temp_state not in frontier_state_list:
                if temp_state not in explored_set.list:
                    temp_path = path + [temp_move]
	            frontier.push(( temp_state, temp_path ))

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    #Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]

    frontier = util.PriorityQueue() # LIFO  ( (x-y_coodinate ,path to this coordinate, step cost ) , total cost to this coor )
    explored_set = util.Stack() # FIFO (x-y_coordinate)


    # Is the start a goal?
    if problem.isGoalState(problem.getStartState()):
        print "The start node is the goal, program terminating.... "
	return []
    # initial frontier with start state
    frontier.push( (problem.getStartState(), [], 0 ), 0 )
    while True:
        # check if the frontier is empty
        if frontier.isEmpty():
            print "The frontier is empty, program terminating.... "
            return []
       
        # choose one node to expand
	half_explore = frontier.pop()
        state = half_explore[0]
        path = half_explore[1]
        cost = half_explore[2]

        # check if the selected_node is the goal
        if problem.isGoalState(state):
	    return path
    
        explored_set.push(state)
        #### relaxation #### 
        for successor in problem.getSuccessors(state):
            temp_state = successor[0]
            temp_move = successor[1]
            step_cost = successor[2]
            frontier_state_list = []
            for node in frontier.heap:
                frontier_state_list.append( node[2][0] )
	    if temp_state not in frontier_state_list:
                if temp_state not in explored_set.list:
                    temp_path = path + [temp_move]
                    temp_cost = cost + step_cost
	            frontier.push( (temp_state, temp_path, temp_cost), temp_cost )

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue() # LIFO  ( (x-y_coodinate ,path to this coordinate, step cost ) , total cost to this coor )
    explored_set = util.Stack() # FIFO (x-y_coordinate)


    # Is the start a goal?
    if problem.isGoalState(problem.getStartState()):
        print "The start node is the goal, program terminating.... "
	return []
    # initial frontier with start state
    frontier.push( (problem.getStartState(), []), 0 )
    while True:
        # check if the frontier is empty
        if frontier.isEmpty():
            print "The frontier is empty, program terminating.... "
            return []
       
        # choose one node to expand
	half_explore = frontier.pop()
        state = half_explore[0]
        path = half_explore[1]

        # check if the selected_node is the goal
        if problem.isGoalState(state):
	    return path
    
        explored_set.push(state)
        #### relaxation #### 
        for successor in problem.getSuccessors(state):
            temp_state = successor[0]
            heuristicCost = heuristic(temp_state, problem)
            temp_move = successor[1]
            frontier_state_list = []
            for node in frontier.heap:
                frontier_state_list.append(node[2][0])
	    if temp_state not in frontier_state_list:
                if temp_state not in explored_set.list:
                    temp_path = path + [temp_move]
                    temp_cost = problem.getCostOfActions(temp_path) +  heuristicCost 
	            frontier.push( (temp_state, temp_path), temp_cost )
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
