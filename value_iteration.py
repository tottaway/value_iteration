"""
Implementation of value iteration algorithm for MDPs using Russell and Norwigg
as reference.
"""

import numpy as np
import matplotlib.pyplot as plt

class Agent():
    def __init__(self, states, terminal_states, actions, transition, reward):
        self.states = states
        self.terminal_states = terminal_states

        self.actions = actions

        self.transition = transition

        self.reward = np.copy(reward)
        self.U = np.copy(reward)

        self.Us = []
        self.value_iteration()

    def value_iteration(self):
        U_prime = np.zeros_like(self.U)

        for i, s in enumerate(self.states):
            if s in self.terminal_states:
                U_prime[i] = self.reward[i]
                continue
            best_action = None
            best_util = -1e9 # -inf
            for a in actions:
                total_util = 0
                for j, next_s in enumerate(self.states):
                    weight = self.transition(s, a, next_s)
                    util = self.U[j]
                    total_util += weight * (self.reward[i] + util)

                if total_util > best_util:
                    best_action = a
                    best_util = total_util

            U_prime[i] = best_util

        change = np.abs(self.U - U_prime).sum()
        thresh = 1e-15
        self.Us.append(np.copy(self.U))
        self.U = U_prime
        if change > thresh:
            self.value_iteration()

    def graph(self):
        plt.plot(np.arange(0, len(self.Us)), self.Us)
        plt.show()

    def policy(self, state):
        if state in self.terminal_states:
            return "terminal"

        best_action = None
        best_action_val = 0
        for a in self.actions:
            action_val = 0
            for j, next_s in enumerate(self.states):
                weight = self.transition(state, a, next_s)
                val = self.U[j]
                action_val += weight * val
            if action_val > best_action_val:
                best_action = a
                best_action_val = action_val
        return best_action


        
def optimal_policy(state):
    if state == (1, 1):
        return "move-up"
    elif state == (1, 2):
        return "move-up"
    elif state == (1, 3):
        return "move-right"
    elif state == (2, 1):
        return "move-left"
    elif state == (2, 2):
        return "move-right"
    elif state == (2, 3):
        return "move-right"
    elif state == (3, 1):
        return "move-left"
    elif state == (3, 2):
        return "move-up"
    elif state == (3, 3):
        return "move-right"
    elif state == (4, 1):
        return "move-left"
        

def grid_transition(state, action, next_state):
    """This is the transition funciton for the gridworld show on page 501 of
    the Russell and Norvig book"""

    terminal_states = [(4, 2), (4, 3)]
    if state in terminal_states:
        raise ValueError("Taking actions in the terminal state")

    potential_next_states = {
        "move-up": (state[0], state[1] + 1),
        "move-left": (state[0] - 1, state[1]),
        "move-right": (state[0] + 1, state[1]),
        "move-down": (state[0], state[1] - 1),
    }
    # handle bumping into walls
    bad_squares = [(2, 2)]
    for k, v in potential_next_states.items():
        if (1 > v[0] or
            4 < v[0] or
            1 > v[1] or
            3 < v[1]) or v in bad_squares:
            potential_next_states[k] = state

    if next_state == potential_next_states[action]:
        return 0.8
    elif action == "move-up" or action == "move-down":
        if (next_state == potential_next_states["move-left"] or
            next_state == potential_next_states["move-right"]):
            if potential_next_states["move-left"] == potential_next_states["move-right"]:
                return 0.2
            return 0.1
    elif action == "move-left" or action == "move-right":
        if (next_state == potential_next_states["move-up"] or
            next_state == potential_next_states["move-down"]):
            if potential_next_states["move-up"] == potential_next_states["move-down"]:
                return 0.2
            return 0.1

    return 0

if __name__ == "__main__":
    states = [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 1),
        (4, 2),
        (4, 3)
    ]

    grid_reward = [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -1., 1.]
    state_to_reward = {s: r for s, r in zip(states, grid_reward)}

    actions = [
        "move-up",
        "move-left",
        "move-right",
        "move-down"
    ]
    terminal_states = [(4, 2), (4, 3)]
    agent = Agent(states, terminal_states, actions, grid_transition, grid_reward)
    
    for i, s in enumerate(states):
        print(s, agent.policy(s), agent.U[i])
