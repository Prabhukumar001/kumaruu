import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Simulate traffic data for intersections
def simulate_traffic_data(num_intersections, num_time_steps):
    data = {
        'intersection_id': [],
        'time_step': [],
        'vehicle_count': [],
        'average_speed': []
    }
    for i in range(num_intersections):
        for t in range(num_time_steps):
            data['intersection_id'].append(i)
            data['time_step'].append(t)
            data['vehicle_count'].append(np.random.randint(20, 100))  # Simulate vehicle count
            data['average_speed'].append(np.random.uniform(10, 40))   # Simulate speed in km/h
    return pd.DataFrame(data)

# Initialize traffic data simulation
num_intersections = 5
num_time_steps = 100  # Define simulation duration in time steps
traffic_data = simulate_traffic_data(num_intersections, num_time_steps)

# Step 2: Analyze traffic patterns using Linear Regression
X = traffic_data[['time_step']]
y = traffic_data['vehicle_count']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 3: Define Q-learning-based Traffic Signal Agent
class TrafficSignalAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        # Epsilon-greedy choice between exploration and exploitation
        if np.random.rand() < self.epsilon:  
            return np.random.randint(0, self.q_table.shape[1])
        else:  
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

# Define states and actions for Q-learning agent
num_states = 10  # Discretized traffic density levels
num_actions = 2  # Actions: 0 = Keep light as is, 1 = Change light

# Initialize Q-learning agent
agent = TrafficSignalAgent(num_states, num_actions)

# Train the agent through simulated experience
for _ in range(1000):  # Training episodes
    state = np.random.randint(0, num_states)
    action = agent.choose_action(state)
    next_state = np.random.randint(0, num_states)
    reward = 1 if next_state < state else -1  # Reward for reducing traffic density
    agent.update_q_value(state, action, reward, next_state)

# Step 4: Real-time traffic signal control based on agent's decision
def control_traffic_signal(agent, current_density):
    state = int(current_density / 10)  # Convert density to discrete state
    action = agent.choose_action(state)
    if action == 1:
        print("Changing traffic light to alleviate congestion.")
    else:
        print("Keeping current traffic light setting.")

# Example of real-time traffic control
current_density = 45  # Simulated real-time traffic density
predicted_density = model.predict(pd.DataFrame([[current_density]], columns=["time_step"]))[0]
print(f"Predicted vehicle count at current time: {predicted_density:.2f}")
control_traffic_signal(agent, current_density)