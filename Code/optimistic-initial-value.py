# Optimistic Initial Values ####################################################

# Pseudo Code

# Initialize bandit means to some large value
# While TRUE:
#   p = random no in [0, 1]
#   if p < epsilon:
#       j = choose a random bandit
#   else:
#       j = argmax(estimated bandit means)
#   x = play bandit j and gain reward bandits[j]
#   update estimated mean of j
#   update epsilon if cooling schedule specified

import numpy as np

class Bandit_OIV:
    # Initialize real value of p and estimate of p to large value
    def __init__(self, p):
        self.p = p
        self.p_estimate = 10
        self.N = 0

    # Simulation- generate win with probability p- produces a TRUE/FALSE Boolean
    def pull(self):
        return(np.random.random() < self.p)

    # Update estimated probability of success
    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N


# Carry out simulation
