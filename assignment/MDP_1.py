# %%
import numpy

# %% [markdown]
#     Environment setup

# %%


class Environment:
    def __init__(self):
        self.nS = 16
        self.nA = 4
        self.states = [i for i in range(16)]
        self.actions = {0: 'Left', 1: 'Up', 2: 'Right', 3: 'Down'}
        self.env = {
            state: {
                action: {
                    's_Prob': 0, 'n_State': 0, 's_Reward': 0, 'Terminated': False
                } for action in [i for i in range(4)]
            } for state in [i for i in range(16)]
        }
        states_set = set(range(16))
        action = 0  # action Left
        for state in [0, 4, 8, 12]:
            self.env[state][0] = {
                's_Prob': 1, 'n_State': state, 's_Reward': -5, 'Terminated': False}

        for state in states_set - set([0, 4, 8, 12]):
            self.env[state][0] = {
                's_Prob': 1, 'n_State': state-1, 's_Reward': -1, 'Terminated': False}

        action = 1  # Action Up
        for state in [0, 1, 2, 3]:
            self.env[state][1] = {
                's_Prob': 1, 'n_State': state, 's_Reward': -5, 'Terminated': False}

        for state in states_set - set([0, 1, 2, 3]):
            self.env[state][1] = {
                's_Prob': 1, 'n_State': state-4, 's_Reward': -1, 'Terminated': False}

        action = 2  # right
        for state in [3, 7, 11]:
            self.env[state][2] = {
                's_Prob': 1, 'n_State': state, 's_Reward': -5, 'Terminated': False}

        for state in states_set - set([3, 7, 11, 15]):
            self.env[state][2] = {
                's_Prob': 1, 'n_State': state+1, 's_Reward': -1, 'Terminated': False}

        action = 3  # action Down
        for state in [12, 13, 14]:
            self.env[state][3] = {
                's_Prob': 1, 'n_State': state, 's_Reward': -5, 'Terminated': False}

        for state in states_set - set([12, 13, 14, 15]):
            self.env[state][3] = {
                's_Prob': 1, 'n_State': state+4, 's_Reward': -1, 'Terminated': False}

        state = 15
        self.env[state][0] = {'s_Prob': 1, 'n_State': state,
                              's_Reward': 0, 'Terminated': True}  # left
        self.env[state][1] = {'s_Prob': 1, 'n_State': state,
                              's_Reward': 0, 'Terminated': True}  # up
        self.env[state][2] = {'s_Prob': 1, 'n_State': state,
                              's_Reward': 0, 'Terminated': True}  # right
        self.env[state][3] = {'s_Prob': 1, 'n_State': state,
                              's_Reward': 0, 'Terminated': True}  # down


# %%
def reward_funtion(prob_matrix, envirnoment):
    state_reward = [[envirnoment.env[state][action]['s_Reward']
                     for action in range(4)] for state in range(16)]
    # print(state_reward)
    reward = state_reward * prob_matrix
    # print(reward)
    return reward

# %%


def value_funtion(environment, values, reward, prob_matrix, gamma):   
    values = reward + gamma * (prob_matrix * reward)
    return values


# %% [markdown]
#     main funtion

# # %%
# def main():
#     environment = Environment()
#     print("Environment --------")
#     print(f'\nnumber of states : {environment.nS}')
#     print(f'number of action at each state : {environment.nA}')
#     print(f'states {environment.states}')
#     print(f'action {environment.actions.values()}')
#     print('\n\n')
#     # print(f'env {environment.env}')
#     prob_matrix_states = [[environment.env[state][action]['s_Prob']
#                            for action in range(4)] for state in range(16)]
#     # print(prob_matrix)

#     # random uniform policy in staring
#     action_prob = numpy.ones((environment.nS, environment.nA)) / environment.nA
#     print(f"policy : \n {action_prob}")
#     # prob matrix to reach state {s'} from {s} given action 'a'
#     prob_matrix = prob_matrix_states * action_prob
#     # print(prob_matrix)
#     # print(prob_matrix.shape)
#     reward = reward_funtion(prob_matrix=prob_matrix, envirnoment=environment)
#     print(f"reward : \n {reward}")

#     values = numpy.zeros((environment.nS, environment.nA))

#     values = value_funtion(environment=environment, values=values,
#                            reward=reward, prob_matrix=prob_matrix, gamma=1)

#     print(f"values : \n {values}")


# # %%
# if __name__ == '__main__':
#     main()

# # %%
# environment = Environment()
# print(environment.env)

# # %%
