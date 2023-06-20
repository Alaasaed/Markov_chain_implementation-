import numpy as np

# Get the transition probability matrix P from the user
try:
    n_states = int(input("Enter the number of states: "))
    P = np.zeros((n_states, n_states))
    for i in range(n_states):
        row = input("Enter the transition probabilities for state " + str(i) + ": ")
        P[i, :] = [float(x) for x in row.split()]
except ValueError:
    print("Invalid input. Please enter numeric values.")

# Get the initial probability vector pi0 from the user
try:
    pi0_input = input("Enter the initial state probabilities: ")
    pi0 = np.array([float(x) for x in pi0_input.split()])
except ValueError:
    print("Invalid input. Please enter numeric values.")

# Get the number of steps n from the user
try:
    n = int(input("Enter the number of steps: "))
except ValueError:
    print("Invalid input. Please enter an integer value.")

# Get the state ID from the user
try:
    state_id = int(input("Enter the state ID: "))
except ValueError:
    print("Invalid input. Please enter an integer value.")

# Check if Markov chain is irreducible
is_irreducible = True
for i in range(P.shape[0]):
    if np.sum(P[i, :]) == 0:
        is_irreducible = False
        break

# Check if Markov chain is aperiodic
is_aperiodic = True
periods = np.zeros(n_states)
for i in range(n_states):
    visited = set()
    current = i
    period = 0
    while current not in visited:
        visited.add(current)
        current = np.argmax(P[current])
        period += 1
    periods[i] = period
    if period != 1:
        is_aperiodic = False

# Check if Markov chain is absorbing
is_absorbing = False
if np.any(np.diag(P) == 1):
    is_absorbing = True

# Determine the type of Markov chain class
markov_chain_class = ""
if is_absorbing:
    markov_chain_class = "Absorbing"
elif is_irreducible and is_aperiodic:
    markov_chain_class = "Irreducible and Aperiodic"
elif is_irreducible and not is_aperiodic:
    markov_chain_class = "Irreducible and Periodic"
elif not is_irreducible and is_aperiodic:
    markov_chain_class = "Reducible and Aperiodic"
else:
    markov_chain_class = "Reducible and Periodic"

print("The Markov chain class is", markov_chain_class)

# Check type of state
state_type = ""
if np.sum(P[:, state_id]) > 0 and np.sum(P[state_id, :]) > 0:
    state_type = "Transient"
elif np.sum(P[:, state_id]) > 0 and np.sum(P[state_id, :]) == 0:
    state_type = "Absorbing"
else:
    state_type = "Isolated"

print("The type of state", state_id, "is", state_type)

# Find steady state
eigenvalues, eigenvectors = np.linalg.eig(np.transpose(P))
steady_state = eigenvectors[:, np.isclose(eigenvalues, 1.0)]
steady_state = steady_state[:, 0] / np.sum(steady_state[:, 0])

print("The steady state probabilities are", steady_state)

# State probabilities after n steps
state_probs = np.matmul(np.linalg.matrix_power(P, n), pi0)

print("The state probabilities after", n, "steps are", state_probs)






