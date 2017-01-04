# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.Q_values = util.Counter() # A Counter is a dict with default 0


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.Q_values[(state, action)]    

        #util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return float(0)
        else:
            Q_list = []
            for action in legalActions:
                Q_list.append( self.getQValue(state, action) )
            best_Q = max(Q_list)
            return best_Q	
             
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # should break ties randomly for better behavior. The random.choice() function will help. 
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        else:
            best_Q = float(-99999999)
            best_action = -1
            best_action_list = []
            for action in legalActions:
                temp_Q = self.getQValue(state, action)
                if temp_Q > best_Q:
                    best_Q = temp_Q
                    best_action = action
                    best_action_list = []
                    best_action_list.append(best_action)
                elif temp_Q == best_Q:
                    best_action_list.append(action)


            if len(best_action_list) > 1:  # if multiple actions tie in value 
                best_action = random.choice(best_action_list) 

            return best_action

        # NO BREAK TIE 
        '''
        max_action = None
        max_q_val = 0
        for action in self.getLegalActions(state):
            q_val = self.getQValue(state, action)
            if q_val > max_q_val or max_action is None:
                max_q_val = q_val
                max_action = action
        return max_action
        '''

        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
            return None     
        if util.flipCoin(self.epsilon):  # epsilon prob to randomly pick an action 
            action = random.choice(legalActions)
            return action
        else:   # return the best policy action
            action = self.computeActionFromQValues(state)
            return action
             
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        best_Q = float(0)
        # if there are no legal action in nextState
        if len(self.getLegalActions(nextState))==0:
            best_Q = float(0)
        else:
            Q_list = []
            for next_action in self.getLegalActions(nextState):
                Q_list.append( self.getQValue(nextState, next_action) )
            best_Q = max(Q_list)
                #temp_Q = self.getQValue(nextState, next_action)
                #if temp_Q > best_Q:
                #    best_Q = temp_Q
                     
        new_Q = self.getQValue(state, action)  + self.alpha * ( reward + self.discount * best_Q - self.getQValue(state, action) )
        self.Q_values[(state,action)] = new_Q
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        weights = self.weights
        features = self.featExtractor.getFeatures(state,action)
        Q_value = 0.0
        for feature in features:
            Q_value += features[feature] * weights[feature]
        return Q_value 
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        differece = float(0)
        best_Q = float(-99999999)
        if len(self.getLegalActions(nextState))==0:
            best_Q = float(0)
        else:
            for next_action in self.getLegalActions(nextState):
                temp_Q = self.getQValue(nextState, next_action)
                if temp_Q > best_Q:
                    best_Q = temp_Q
        
        difference = reward + self.discount * best_Q - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state,action)
        for feature in features:
            self.weights[feature] = self.weights[feature] + self.alpha * difference * features[feature]


        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
