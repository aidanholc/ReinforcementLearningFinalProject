from pokerkit import *
import numpy as np
from tqdm import tqdm
import copy
from concurrent.futures import ProcessPoolExecutor

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from frozendict import frozendict
from collections import deque

# Hardcode possible actions
ACTIONS = ['fold','check_or_call','quarter_pot','half_pot','pot','two_pot','all_in']

class BaseAgent:
    def __init__(self, config):
        self.config = config
    
    def clean_action(self, raw_action, state, min_bet_size, can_raise_func, can_fold):
        """
        Clean's agent's action into something the environment can handle
        """               
        action_dict = {'fold':False,'check_or_call':False,'raise':False,'bet_size':-999}
        if type(raw_action) is list:
            raw_action = raw_action[0]
        match raw_action:
            case 'fold':
                if can_fold:
                    action_dict['fold'] = True
                else:
                    action_dict['check_or_call'] = True
                    raw_action = 'check_or_call'
                return action_dict, raw_action
            case 'check_or_call':
                action_dict['check_or_call'] = True
                return action_dict, raw_action
            case 'quarter_pot':
                bet_size = min(max(state[1]*.25, min_bet_size), state[0]) # Index 1 is pot size, index 0 is stack
                if can_raise_func(bet_size):
                    action_dict['raise'] = True
                    action_dict['bet_size'] = bet_size
                else:
                    action_dict['check_or_call'] = True
                return action_dict, raw_action
            case 'half_pot':
                bet_size = min(max(state[1]*.5, min_bet_size), state[0])
                if can_raise_func(bet_size):
                    action_dict['raise'] = True
                    action_dict['bet_size'] = bet_size
                else:
                    action_dict['check_or_call'] = True
                return action_dict, raw_action
            case 'pot':
                bet_size = min(max(state[1], min_bet_size), state[0])
                if can_raise_func(bet_size):
                    action_dict['raise'] = True
                    action_dict['bet_size'] = bet_size
                else:
                    action_dict['check_or_call'] = True
                return action_dict, raw_action
            case 'two_pot':
                bet_size = min(max(2*state[1], min_bet_size), state[0])
                if can_raise_func(bet_size):
                    action_dict['raise'] = True
                    action_dict['bet_size'] = bet_size
                else:
                    action_dict['check_or_call'] = True
                return action_dict, raw_action
            case 'all_in':
                bet_size = state[0]
                if can_raise_func(bet_size):
                    action_dict['raise'] = True
                    action_dict['bet_size'] = bet_size
                else:
                    action_dict['check_or_call'] = True
                return action_dict, raw_action
            case _:
                raise Exception(f"Passed in bad raw action: {raw_action}")


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)    #First fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)    #Second fully connected layer
        self.fc3 = nn.Linear(hidden_size, 1)    #Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))    #ReLU activation for the first layer
        x = F.relu(self.fc2(x))    #ReLU activation for the second layer
        x = self.fc3(x)    #Output layer (no activation because this is a regression problem)
        return x

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size) #using a deque instead of a list should speed up computation as popping at index 0 many times is computationally expensive for a list
            # Adding maxlen for deque also automatically handles popping when full
    def append(self, transition):
        self.buffer.append(transition)
    def batch(self, batch_size):
        batch_size = min(len(self.buffer), batch_size) # make sure we aren't sampling more than there is in buffer
        ret_batch = np.random.choice(self.buffer, batch_size)
        return ret_batch, np.arange(len(ret_batch))
    def extend(self, transitions):
        for transition in transitions:
            self.buffer.append(transition)

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, beta, epsilon):
        self.buffer = deque(maxlen=buffer_size) # Using a deque instead of a list should speed up computation as popping at index 0 many times is computationally expensive for a list
                                                # Adding maxlen for deque also automatically handles popping when full
        self.priorities = deque(maxlen=buffer_size)
        self.beta = beta
        self.epsilon = epsilon

    def append(self, transition):
        self.buffer.append(transition)
        #print(self.priorities)
        self.priorities.append(max(self.priorities, default=1)) #give it a high priority so it new things transitions will get trained on
    
    def extend(self, transitions):
        for transition in transitions:
            self.buffer.append(transition)
            self.priorities.append(max(self.priorities, default=1))

    def batch(self, batch_size):
        batch_size = min(len(self.buffer), batch_size) # make sure we aren't sampling more than there is in buffer
        
        #turn priority to probability
        probs = np.array(self.priorities) ** self.beta
        probs = probs / sum(probs) 

        batch_indices = np.random.choice(np.arange(len(self.buffer)), batch_size, p=probs)
        batch = np.array(self.buffer)[batch_indices]
        return batch, batch_indices

    def set_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = td_error + self.epsilon

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class DQNAgent(BaseAgent):
    '''A class to manage the agent'''
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def __init__(self, config):
        '''Set up the constructor
            Takes -- config, a dictionary specifying the track dimensions and initial state
        '''
        self.input_size = 14 # Size of state space
        #self.output_size = 7 # Size of action space
        self.config = config
        self.Q = MLP(input_size=self.input_size, hidden_size=self.config['hidden_size'])
        #self.Q.to('cuda')
        self.Q.train()    #Set the model to training mode
        self.Q_prime = copy.deepcopy(self.Q)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.config['alpha'])
        if config['prioritizedReplay']:
            self.D = PrioritizedReplayBuffer(buffer_size=config['M'], beta=config['beta'], epsilon=config['pr_epsilon'])    #init the replay buffer
        else:
            self.D = ReplayBuffer(buffer_size=config['M'])

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def Q_reset(self):
        '''A function reset the MLP to random initial parameters'''
        self.Q = MLP(input_size=self.input_size, hidden_size=self.config['hidden_size'])
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def update_Q_prime(self):
        '''A function set the target approximator to the online network'''
        self.Q_prime = copy.deepcopy(self.Q)
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def make_options(self, s_t):
        '''A function to create the state action pairs
            Takes:
                s_t -- a list of the state information
            Returns:
                a torch tensor with the first six columns the state information and the last two columns the actions
        '''
        s_tA = []    #init a list to hold the state action information
        for a in self.config['A']:    #loop over actions
            # One hot enode action list
            action_idx = ACTIONS.index(a)
            action_one_hot = list(np.zeros(shape=(len(ACTIONS),)))
            action_one_hot[action_idx] = 1
            #print(f" s_t: {s_t}, a: {a}")
            s_tA.append(s_t + action_one_hot)    #add and record
            #print(f'action_one_hot: {action_one_hot}\n state: {s_t}\nstate action list: {s_tA}')
        return torch.tensor(s_tA).to(torch.float32)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def epsilon_t(self, count, n_episodes):
        '''Lets try out a dynamic epsilon
            Takes:
                count -- int, the number of turns so far
            Returns:
                float, a value for epsilon
        '''
        if count <= self.config['epsilon_burnin']:    #if we're still in the initial period...
            return 1    #choose random action for sure
        else:
            return 1/(n_episodes**0.5)    #otherwise reduce the size of epsilon

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def policy(self, s_t, min_bet_size, can_raise_func, can_fold):
        '''A function to choose actions using Q-values
            Takes:
                s_t -- a torch tensor with the first six columns the state information and the last two columns the actions
                epsilon -- the probability of choosing a random action
        '''
        #print(f'state: {s_t}')
        if np.random.uniform() < 0.2:    #if a random action is chosen...
            return self.clean_action(self.config['A'][np.random.choice(a = range(len(self.config['A'])))],s_t,min_bet_size, can_raise_func, can_fold)    #return the random action
        else:
            return self.clean_action(self.config['A'][torch.argmax(self.Q(self.make_options(s_t)))],s_t, min_bet_size, can_raise_func, can_fold)    #otherwise return the action with the highest Q-value as predicted by the MLP
        
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def make_batch(self):
        '''A function to make a batch from the memory buffer and target approximator
            Returns:
                a list with the state-action pair at index 0 and the target at index 1
        '''
        batch, batch_indices = self.D.batch(batch_size=self.config['B']) #only need batch_indices if using prioritized replay
        if self.config['prioritizedReplay']:
            td_errors, indices = [], []
       # batch = np.random.choice(self.D,self.config['B'])    #sample uniformly
        X,y = [],[]    #init the state-action pairs and target
        for d, i in zip(batch, batch_indices):    #loop over all the data collected
            X.append(d['d_s_a'])    #record the state action pair
            y_t = d['r_t+1']    #compute the target
            if not d['done']:    #if this state didn't end the episode...
                state_action_tensor = self.make_options(d['s_t+1'])
                if self.config['DDQN']:
                    best_online_action = state_action_tensor[np.argmax(self.Q(state_action_tensor).detach().numpy())] # Find best action from online network
                    max_a_Q = float(self.Q_prime(best_online_action)) # Evaluate in target approximator
                else:
                    max_a_Q = float(max(self.Q_prime(state_action_tensor)))    #compute the future value using the target approximator
                y_t = y_t + (self.config['gamma'] ** self.config['multi_step'])*max_a_Q    #update the target with the future value

                if self.config['prioritizedReplay']: # get temporal difference error
                    td_error = abs(y_t - self.Q_prime(torch.tensor(d['d_s_a']).to(torch.float32)).item()) # calculate td error
                    #print(td_error)
                    td_errors.append(td_error)
                    indices.append(i)
            y.append(y_t)    #record the target   
        if self.config['prioritizedReplay']:
            self.D.set_priorities(indices, td_errors)
        return [X,y]

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def update_Q(self,X,y):
        '''A function to update the MLP
            Takes:
                X -- the features collected from the replay buffer
                y -- the targets
        '''
        #do the forward pass
        X = torch.tensor(X).to(torch.float32)
        y = torch.tensor(y).to(torch.float32).view(len(y),1)
        outputs = self.Q(X)    #pass inputs into the model (the forward pass)
        loss = self.criterion(outputs,y)    #compare model outputs to labels to create the loss

        #do the backward pass
        self.optimizer.zero_grad()    #zero out the gradients    
        loss.backward()    #compute gradients
        self.optimizer.step()    #perform a single optimzation step

class FixedPolicyAgent(BaseAgent):
    def __init__(self, config):
        if 'loose_handedness' not in config.keys():
            raise Exception('Must specify loose-handedness in config')
        else:
            self.target_lh = config['loose_handedness']
        super().__init__(config)
    
    def policy(self, state, min_bet_size, can_raise_func, can_fold):
        if state[3] <= self.target_lh:
            return self.clean_action('fold', state, min_bet_size, can_raise_func, can_fold)
        
        n_actions = len(self.config['A']) - 1 # do not include fold in count

        raise_bounds = np.linspace(start=self.target_lh, stop=1, num=n_actions, endpoint=True)
        for selection in range(n_actions, 0, -1): # loop through options, checking to see if hand is strong enough
            if state[3] > raise_bounds[selection-1]:
                return self.clean_action(
                    self.config['A'][selection], # the 0th option is 'fold,' which we have decided against
                    state,
                    min_bet_size,
                    can_raise_func,
                    can_fold
                )

class RandomAgent(BaseAgent):
    def policy(self, state, min_bet_size, can_raise_func, can_fold):
        return self.clean_action(np.random.choice(self.config['A']), state, min_bet_size, can_raise_func, can_fold)
