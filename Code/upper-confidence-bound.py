# Upper Confidence Bound (UCB) #################################################

# Pseudo Code ##################################################################

# Initialize by playing each bandit once
# While TRUE:
#   Play j = argmax(Upper Confidence Bound)

# Initialization ###############################################################

import numpy as np

# Bandit Class #################################################################

class Bandit_UCB:
    # Each bandit object should be a probability, p
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0

    # Pull function
    def pull(self):
        return np.random.random() < self.p

    # Update estimate of win rate
    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N