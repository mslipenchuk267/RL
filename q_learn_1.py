import gym, numpy as np, matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1 # decay
DISCOUNT = 0.95 # A weight, a measure of how important we find future action over current actions
EPISODES = 25000

SHOW_EVERY = 2000 # every 500 episodes

# This sets 20 discrete observations (chunks) for the dimension of the os
# Don't have to make DISCRETE_OS_SIZE have same amount of buckets for each observation
DISCRETE_OS_SIZE= [40] * len(env.observation_space.high) # 20 x 20
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5 # epsilon is a measure of how much random action you want to take
			  # the higher the value the higher the chance to take random action
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 # // divides out to integer, never have float

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Create Q Table with random vals
q_table = np.random.uniform(low =- 2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n])) # 20 x 20 Table, every combo of positon and velocity
# episode metrics
ep_rewards = [] # list containig each episodes rewards
# aggr_ep_rewards is a dictionary containing metrics
# ep is the x-axis essentially
# The avg is the trailing avg, so every 500 eps, this wil average over time. Average should go up
# min shows the worst model we had
# max shows what is the best one
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []} 
#helper function to convert continous stats to discrete states
def get_discrerte_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
	episode_reward = 0
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
		episode_reward += reward
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

	ep_rewards.append(episode_reward)

	if not episode % SHOW_EVERY:
		# work on dictionary
		average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:]) # -SHOW_EVERY: means the last 500
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
		aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
		print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc = 4) # lower right corner
plt.show()