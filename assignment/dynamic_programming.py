import numpy as np
import MDP_1 as md


def one_step_lookahead(environment, state, V, discount_factor):
    # Create a vector of dimensionality same as the number of actions

    action_values = np.zeros(environment.nA)
    # print(f"value : {V}")
    for action in range(environment.nA):
        # print(f"action {action}")
        dic_val = environment.env[state][action]
        list_1 = [tuple(dic_val.values())]
        for probability, next_state, reward, terminated in list_1:
            # print(f"next state {next_state}")
            action_values[action] += probability * \
                (reward + discount_factor * V[int(next_state)])

    return action_values


def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):

    # To record the number of iterations after which the evaluation converged
    evaluation_iterations = 1

    # Initialize the value function vector
    V = np.zeros(environment.nS)

    for i in range(int(max_iterations)):

        # For early stopping
        delta = 0

        for state in range(environment.nS):

            # Store the state's new value for this iteration
            v = 0

            # Check for all possible actions
            for action, action_probability in enumerate(policy[state]):

                # Iterate over all possible next states
                dic_val = environment.env[state][action]
                list_1 = [tuple(dic_val.values())]
                for state_probability, next_state, reward, terminated in list_1:

                    # Calculate the expected value
                    v += action_probability * state_probability * \
                        (reward + discount_factor * V[next_state])

            # Maintain the maximum change of value for each state
            delta = max(delta, abs(V[state] - v))

            # Update the state value
            V[state] = v

        # Update the number of iterations
        evaluation_iterations += 1

        # Early stopping
        if(delta < theta):
            print('Policy evaluated in %d iterations' % evaluation_iterations)
            return V


def policy_iteration(environment, discount_factor=1.0, max_iterations=1e9):

    # Initialize the policy with a uniform distribution over the actions for each state
    policy = np.ones((environment.nS, environment.nA)) / environment.nA

    # Store the number of policies evaluated
    evaluated_policies = 1

    for i in range(int(max_iterations)):

        # For Early Stopping
        stable_policy = True

        # Evaluate the current policy
        V = policy_evaluation(policy, environment,
                              discount_factor=discount_factor)

        for state in range(environment.nS):

            # Get the get action so far
            current_action = np.argmax(policy[state])

            # Perform the one-step lookahead to get the action values for the state
            action_values = one_step_lookahead(
                environment, state, V, discount_factor)

            # Get the best action
            best_action = np.argmax(action_values)

            # If the best action for the state changes, the policy is not yet stable
            if(current_action != best_action):
                stable_policy = False

            # Update the policy for the state
            policy[state] = np.eye(environment.nA)[best_action]

        # Increment the number of policies evaluated
        evaluated_policies += 1

        # Early stopping
        if(stable_policy):
            print('Evaluated %d policies.' % evaluated_policies)
            return policy, V


env = md.Environment()
policy_iteration(environment=env)
