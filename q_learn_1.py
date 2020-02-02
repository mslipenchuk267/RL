import gym, numpy as np

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1 # decay
DISCOUNT = 0.95 # A weight, a measure of how important we find future action over current actions
EPISODES = 25000

SHOW_EVERY = 2000 # every 2000 episodes

# This sets 20 discrete observations (chunks) for the dimension of the os
# Don't have to make DISCRETE_OS_SIZE have same amount of buckets for each observation
DISCRETE_OS_SIZE= [20] * len(env.observation_space.high) # 20 x 20
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5 # epsilon is a measure of how much random action you want to take
			  # the higher the value the higher the chance to take random action
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 # // divides out to integer, never have float

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Create Q Table with random vals
q_table = np.random.uniform(low =- 2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n])) # 20 x 20 Table, every combo of positon and velocity

#helper function to convert continous stats to discrete states
def get_discrerte_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
	if episode % SHOW_EVERY == 0:
		print(episode)
		render = True
	else:
		render = False
	discrete_state = get_discrerte_state(env.reset()) # Pass the continous initialized states to discrete
	done = False
	while not done:

		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)
		new_state, reward, done, _ = env.step(action) # returns continous states
		new_discrete_state = get_discrerte_state(new_state)
		if render:
			env.render()
		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]
			new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT + max_future_q)
			# now we update that action that we just took with the new q value
			q_table[discrete_state+(action, )] = new_q 
		elif new_state[0] >= env.goal_position:
			print(f"We made it on episode {episode}")
			q_table[discrete_state + (action, )] = 0 # the reward is no punishment

		discrete_state = new_discrete_state

	# per episode update
	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value

env.close()
