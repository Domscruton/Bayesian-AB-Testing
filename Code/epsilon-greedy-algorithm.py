# Epsilon-Greedy ###############################################################

# Pseudo-code

# While TRUE:
#   p = random no in [0, 1]
#   if p < epsilon:
#       j = choose a random bandit
#   else:
#       j = argmax(predicted bandit means)
#   x = play bandit j and get reward bandits[j]. Update Mean. Alter Epsilon
#   using cooling schedule if specified

import numpy as np

Num_trials = 10000
EPS = 0.1
Bandit_Probabilities = [0.3, 0.35, 0.4]


# Create a Bandit class that initializes each probability in the list of
# probabilities and then simulates a True/False outcome (1/0 in Python) when a
# particular bandit is played and use the update method to update the
# estimated probability
class Bandit:
    def __init__(self, p):
        # p: win/conversion rate
        self.p = p
        self.p_estimate = 0
        self.N = 0

    def pull(self):
        # Draw a win (converted) with probability p
        # (Generates a Boolean or equivalent binary value for win/ loss)
        return (np.random.random() < self.p)

    def update(self, x):
        self.N += 1
        # update the estimate probability of success
        self.p_estimate = (1 / self.N) * ((self.N - 1) * self.p_estimate + x)


def Simulation():
    # Initialize each probability as a Bandit object
    bandits = [Bandit(p) for p in Bandit_Probabilities]

    # Record metrics
    rewards = np.zeros(Num_trials)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])
    print("optimal j:", optimal_j)

    # Run algorithm
    for i in range(Num_trials):

        # Use epsilon-greedy to select next bandit
        if np.random.random() < EPS:
            num_times_explored += 1
            # choose random bandit
            j = np.random.randint(len(bandits))
        else:
            num_times_exploited += 1
            # choose bandit with optimal p.estimate
            j = np.argmax([b.p_estimate for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        # pull arm for bandit with largest sample (generate a 'win' / 'loss')
        x = bandits[j].pull()

        # update reward log
        rewards[i] = x

        # Update the distribution for the bandit we selected
        bandits[j].update(x)


Simulation()

for b in Bandit_Probabilities:
    print("Mean estimate:", b.p_estimate)

# Declining Epsilon (Cooling Schedule) #########################################

# To create a cooling schedule we simply have to alter the value of epsilon at
# the update stage
