# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent, Actions, Grid


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        #legalMoves = [ move for move in legalMoves if move != "Stop" ] ################
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        #newScaredTimes [0, 0]
        #newPos (16, 5)
        #newFood
        """
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        currentFood = state.getFood()
        if currentFood[x][y] == True: ...
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        oldPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  
        #print "g_pos", currentGameState.getGhostPositions
        '''
        "*** YOUR CODE HERE ***"
        ### method 1 ###
        totalScore=0.0
        for ghost in newGhostStates:
            d=manhattanDistance(ghost.getPosition(), newPos)
            factor=1
            if(d<=1):
                if(ghost.scaredTimer!=0):
                  factor=-1
                  totalScore+=2000
                else:
                  totalScore-=200
                  #totalScore-=1000

        for capsule in currentGameState.getCapsules():
            d=manhattanDistance(capsule,newPos)
            if(d==0):
                totalScore+=100
                #totalScore+=500
            else:
                totalScore+=10.0/d
                #totalScore-=10.0/d
          

        for x in xrange(oldFood.width):
            for y in xrange(oldFood.height):
                if(oldFood[x][y]):
                    new_d=manhattanDistance((x,y),newPos)
                    old_d=manhattanDistance((x,y),oldPos)
                    if(new_d==0):
                         totalScore+=100
                    else:
                         totalScore+=1/(new_d*new_d)
        ### END of method 1 ###
        '''
        ### method 2 ###
        '''
        if successorGameState.isWin():
            return float("inf") - 20
        ghostposition = currentGameState.getGhostPosition(1)
        distfromghost = util.manhattanDistance(ghostposition, newPos)
        score = max(distfromghost, 3) + successorGameState.getScore()
        foodlist = newFood.asList()
        closestfood = 100
        for foodpos in foodlist:
            thisdist = util.manhattanDistance(foodpos, newPos)
            if (thisdist < closestfood):
                closestfood = thisdist
        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            score += 100
        if action == Directions.STOP:
            score -= 3
        score -= 3 * closestfood
        capsuleplaces = currentGameState.getCapsules()
        if successorGameState.getPacmanPosition() in capsuleplaces:
            score += 120
        totalScore = score
        '''
        ### END of method 2 ###

        ### method 3 ###
        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        for ghostPos in currentGameState.getGhostPositions():
            if manhattanDistance(newPos, ghostPos) <= 1:
                score -= 999999
        actions = breadthFirstSearch(AnyFoodSearchProblem(successorGameState, oldFood))
        score -= len(actions)
        totalScore = score
        ### END of method 3 ###

        return totalScore
        #return successorGameState.getScore()   # default answer

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        # for AB pruning
        #self.a = int(-100000)
        #self.b = int(100000)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using : 
          (1) self.depth
          (2) self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax :

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        bestScore,bestMove=self.maxFunction(gameState,self.depth)
        #print "bestScore",bestScore
        return bestMove

    
    def maxFunction(self,gameState,depth):
        if depth==0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "noMove"

        moves=gameState.getLegalActions()
        #scores = [self.minFunction(gameState.generateSuccessor(self.index,move),1, depth) for move in moves]
        scores = [self.minFunction(gameState.generateSuccessor(self.index,move),1, depth)[0] for move in moves]
        bestScore=max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0]
        return bestScore,moves[chosenIndex]

    def minFunction(self,gameState,agent, depth):  
        if depth==0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "noMove"
        moves=gameState.getLegalActions(agent) #get legal actions.
        scores=[]
        if(agent!=gameState.getNumAgents()-1):
            #scores =[self.minFunction(gameState.generateSuccessor(agent,move),agent+1,depth) for move in moves]
            scores =[self.minFunction(gameState.generateSuccessor(agent,move),agent+1,depth)[0] for move in moves]
        else:
            scores =[self.maxFunction(gameState.generateSuccessor(agent,move),(depth-1))[0] for move in moves]
        minScore=min(scores)
        worstIndices = [index for index in range(len(scores)) if scores[index] == minScore]
        chosenIndex = worstIndices[0]
        return minScore, moves[chosenIndex]
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        a = int(-100000)
        b = int(100000)
        bestScore,bestMove=self.maxAB(gameState,self.depth, a, b)
        #print "bestScore",bestScore
        return bestMove

    def maxAB(self,gameState,depth, a, b):
        if depth==0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "noMove"
        v_list = []
        best_v = int(-10000)
        best_move = "None"
        #a = self.a
        #b = self.b
        moves = gameState.getLegalActions()
        
        for move in moves:
            action_v = self.minAB(gameState.generateSuccessor(self.index,move),1,depth, a, b)[0]
            if action_v > best_v:
                best_v = action_v
                best_move = move  
            if best_v>b or best_v==b:  # pruning happens, so no need to append to valuelist
                return best_v, best_move
            a = max(a , best_v)
            # self.a = a
        return best_v, best_move


    def minAB(self,gameState,agent, depth, a, b):  
        if depth==0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "noMove"
        moves=gameState.getLegalActions(agent) #get legal actions.
        scores=[]
        v_list = []
        worst_v = int(10000)
        worst_move = "None"
        #a = self.a
        #b = self.b 
        if(agent!=gameState.getNumAgents()-1):
            for move in moves:
                action_v = self.minAB( gameState.generateSuccessor(agent,move),agent+1,depth, a, b)[0] 
                if action_v < worst_v :
                    worst_v = action_v
                    worst_move = move
                if a>worst_v or a==worst_v: # worst_v is used to cut down b, this line means b < a, pruning happens !
                    return worst_v, worst_move
                b = min(b, worst_v)
                #self.b = b 
        else:
            for move in moves:
                action_v = self.maxAB(gameState.generateSuccessor(agent,move),(depth-1), a, b)[0] 
                
                if action_v < worst_v :
                    worst_v = action_v
                    worst_move = move
                if a>worst_v or a==worst_v:
                    return worst_v, worst_move
                b = min(b, worst_v)
                #self.b = b 
        return worst_v, worst_move
        #util.raiseNotDefined()



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestScore,bestMove=self.maxFunction(gameState,self.depth)
        return bestMove

    
    def maxFunction(self,gameState,depth):
        if depth==0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "noMove"

        moves=gameState.getLegalActions()
        scores = [self.EXPFunction(gameState.generateSuccessor(self.index,move),1, depth) for move in moves]
        #scores = [self.EXPFunction(gameState.generateSuccessor(self.index,move),1, depth) for move in moves if move != "Stop"]
        bestScore=max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0]
        return bestScore,moves[chosenIndex]

    def EXPFunction(self,gameState,agent, depth):  
        if depth==0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        moves=gameState.getLegalActions(agent) #get legal actions.
        scores=[]
        if(agent!=gameState.getNumAgents()-1):
            scores = [self.EXPFunction(gameState.generateSuccessor(agent,move),agent+1,depth) for move in moves ]
            #scores = [self.EXPFunction(gameState.generateSuccessor(agent,move),agent+1,depth) for move in moves if move!="Stop"]
        else:
            scores = [self.maxFunction(gameState.generateSuccessor(agent,move),(depth-1))[0] for move in moves ]
            #scores = [self.maxFunction(gameState.generateSuccessor(agent,move),(depth-1))[0] for move in moves if move!="Stop"]
        scores = [ float(score) for score in scores ]
        EXPscore = sum(scores) / float(len(scores))
        return float(EXPscore)
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    #  The evaluation function should evaluate states
    #  METHOD 1 : depth-limit DFS and keep track. Reward and penalty ACCUMULATED !
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      Design methodology : 
      (1) The main breakthrough is to apply BFS on Anyfoodsearch problem (from the seach project), which returns a optimal shortest path to all foods. If state gets longer path, it gets worse value. 
      (2) Then we consider the ghost status. If ghost is scared, we encourage the pacman to get closer to them. Eating them get really big value.
      (3) But before, we make the ghost scared, we must first eat the Capsules. So the same we encourage the pacman to get closer to capsule. Eating them will get much reward too.
      (4) One exciting observation : When a pacman is near the capsule and the ghosts are getting closer, pacman will wait for the ghosts to come closer then eat the capsule then eat the ghost. Like a patient hunter, very interesting ! 
    """
    #successorGameState = currentGameState.generatepacmansuccessor(action)
    #oldPos = currentGameState.getPacmanPosition()
    #newFood = successorGameState.getFood()
    #newGhostStates = successorGameState.getGhostStates()
    #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  
    # newScaredTimes [0, 0]
    # newPos (16, 5)
    # newFood
    # Useful information you can extract from a GameState (pacman.py)

    """
    Being far away from ghosts is better if pacman is not invincible
    Being close to ghosts is better if pacman is invincible
    Having less regular dots left to collect on the board is better
    Having more "invincible dots" still remaining on the board is better
    Having more ghosts in prison is better
    """

    # Method 1 : peek for depth away state
    #depth = 2
    #score = BFS(currentGameState, depth)
    # End of Method1 

    # Method 2
    total_score_list = []
    oldFood = currentGameState.getFood()  
    actions = currentGameState.getLegalActions()

    '''
    for action in actions:
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        score = successorGameState.getScore()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        #for ghostPos in currentGameState.getGhostPositions():
        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) <= 1:
                if (ghost.scaredTimer!=0):
                    score += 2000
                else:
                    score -= 999999
        least_moves = breadthFirstSearch(AnyFoodSearchProblem(successorGameState, oldFood))
        score -= len(least_moves)

        # extra point for Capsule
        for capsule in currentGameState.getCapsules():
            d = manhattanDistance(capsule,newPos)
            if(d==0):
                score+=100
            #else:
                #totalScore+=10.0/d

        #new food num < old food num : critical break local optimal !!!
        if len(newFood.asList()) < len(oldFood.asList()):
            score+=50

        total_score_list.append(float(score))

    if len(total_score_list)==0:
        #print "There are no legal moves, this shuold never happen !!!!"
        return -99999999999
    
    total_score = float(max(total_score_list))
    #total_score = sum(total_score_list) / len(total_score_list)
    ### END of method 2 ###
    '''

    
    # Method 3
    total_score_list = []
    oldFood = currentGameState.getFood()  
    actions = currentGameState.getLegalActions()

    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    now_Pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    #newFood = successorGameState.getFood()
    #newGhostStates = successorGameState.getGhostStates()
    now_GhostStates = currentGameState.getGhostStates()

    total_ghost_d = 0
    for ghost in now_GhostStates:
        if manhattanDistance(now_Pos, ghost.getPosition()) <= 1:
            if (ghost.scaredTimer!=0):
                score += 2000
            else:
                score -= 999999
        # eager to eat ghost if ghost is scared !
        if (ghost.scaredTimer!=0):
            total_ghost_d  += manhattanDistance(now_Pos, ghost.getPosition())
    if total_ghost_d != 0:
        score += 10/float(total_ghost_d)

    least_moves = breadthFirstSearch(AnyFoodSearchProblem(currentGameState, oldFood))
    score -= len(least_moves)

    # extra point for Capsule
    for capsule in currentGameState.getCapsules():
        d = manhattanDistance(capsule,now_Pos)
        if(d==0):
            score += 1000
        else:
            score+=100.0/d
    total_score = score
    #new food num < old food num : critical break local optimal !!!
    #if len(newFood.asList()) < len(oldFood.asList()):
    #    score+=50

    #total_score_list.append(float(score))

    #if len(total_score_list)==0:
    #    print "There are no legal moves, this shuold never happen !!!!"
    #    return -99999999999
    
    #total_score = float(max(total_score_list))
    #total_score = sum(total_score_list) / len(total_score_list)
    ### END of method 3 ###



    return total_score # score by betterevaluationfunction

########################################################################################################################
########################################################################################################################
########################################################################################################################

# self-add function for betterevaluationfunction
def BFS(game_state, depth=2):
    # element:(gamestate, path, value)
    for current_depth in range(depth):
        new_history_list = []
        for j in range(len(history_list)):
            gameState = history_list[j][0]
            path = history_list[j][1] 
            actions = gameState.getLegalActions() #get legal actions.
            for action in actions:
                #if action == "Stop":
                #    continue
                successorGameState = gameState.generatePacmanSuccessor(action)
                newPos = successorGameState.getPacmanPosition()
                path = path + [action]
                value = return_Totaldistance_top4food(successorGameState, newPos)
                new_history_list.append( (successorGameState, path, value) )
        history_list = new_history_list
        if current_depth == (depth-1):  
            sorted(new_history_list, key=itemgetter(2))
    #best_path = new_history_list[0][1]
    #best_current_action = best_path[0]
    # Average the final scores of the most outer frontier
    value_list = [ item[2] for item in new_history_list ]
    average_value = sum(value_list) / float(len(value_list))
    return average_value
                
# MST : used by evaluate_final_state(): 
def return_Totaldistance_top4food(newgamestate, pac_position):
    newFood = successorGameState.getFood()
    foodlist = newFood.aslist()
    total_distance = 0
 
    original_len = len(foodlist)
  
    while len(foodlist) != (original_len-4):
        min_dist = 100000
        min_index = -1
        # to find the closest in remaining foods
        for i in range(len(foodlist)):
            temp_dist = mazeDistance(pac_position, foodlist[i], gameState)
            if temp_dist < min_dist:
                min_dist = temp_dist
                min_index = i

        total_distance += min_dist
        pac_position = foodlist[min_index]
        foodlist.remove(foodlist[min_index])

    return float(total_distance)



# TO EVALUATE THE FINAL POS'S VALUE
def evaluate_final_state(): 
    actions=gameState.getLegalActions() #get legal actions.
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    oldPos = currentGameState.getPacmanPosition()
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  
    #print "g_pos", currentGameState.getGhostPositions
    totalScore=0.0
    for ghost in newGhostStates:
        d=manhattanDistance(ghost.getPosition(), newPos)
        factor=1
        if(d<=1):
           if(ghost.scaredTimer!=0):
              factor=-1
              totalScore+=2000
           else:
              totalScore-=200
              #totalScore-=1000

    for capsule in currentGameState.getCapsules():
        d=manhattanDistance(capsule,newPos)
        if(d==0):
            totalScore+=100
            #totalScore+=500
        else:
            totalScore+=10.0/d
            #totalScore-=10.0/d
          

    for x in xrange(oldFood.width):
        for y in xrange(oldFood.height):
            if(oldFood[x][y]):
                new_d=manhattanDistance((x,y),newPos)
                old_d=manhattanDistance((x,y),oldPos)
                if(new_d==0):
                     totalScore+=100
                else:
                     totalScore+=1/(new_d*new_d)
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction



# Self-add function to evaluate (1) state-action pair (2) state  value function
# Import from PROJECT1 : SEARCH PACMAN
class AnyFoodSearchProblem:
  def __init__(self, gameState, goals, start = None, costFn = lambda x: 1):
    self.walls = gameState.getWalls()
    if start == None:
      self.startState = gameState.getPacmanPosition()
    else:
      self.startState = start
    self.costFn = costFn
    self.foods = goals

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
    isGoal = self.foods[state[0]][state[1]]
    return isGoal

  def getSuccessors(self, state):
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append( ( nextState, action, cost) )

    return successors

  def getCostOfActions(self, actions):
    if actions == None: return 999999
    x,y= self.getStartState()
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
      cost += self.costFn((x,y))
    return cost


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue() # LIFO  (x-y_coodinate, path to this coordinate )
    explored_set = util.Stack() # FIFO (x-y_coordinate)


    # Is the start a goal?
    if problem.isGoalState(problem.getStartState()):
        #print "IN BFS : The start node is the goal, program terminating.... "
        return []
    # initial frontier with start state
    frontier.push(( problem.getStartState(),[] ))
    while True:
        # check if the frontier is empty
        if frontier.isEmpty():
            #print "IN BFS : The frontier is empty, program terminating.... "
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


