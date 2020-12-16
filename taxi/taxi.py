import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from IPython.display import clear_output

class QAgent:
    def __init__(self, env, lr, gamma, epsilon):
        self.env = env
        self.counter = 0
        self.learning_rate = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.random.uniform(low = 0.0, high=1.0, size=([500] + [env.action_space.n]))
        self.nsteps = 99
        self.test_print = 0

    def train(self,train_episodes):
        for episode in range(1,train_episodes+1):
            current_state = env.reset()
            steps = 0
            penalties = 0
            done = False
            while not done and steps<self.nsteps:
                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[current_state])
                    
                new_state, reward, done, _ = env.step(action)
                
                current_q = np.max(self.q_table[current_state])
                max_future_q = np.max(self.q_table[new_state])
                new_q = (1-self.learning_rate)*current_q + learning_rate * (reward + self.gamma * max_future_q)
                self.q_table[current_state, action] = new_q

                current_state = new_state
                steps += 1
                if reward == -10:
                    penalties += 1
            self.epsilon *= 0.99
            print("Episode {}: we made it in {} steps with {} penalties".format(episode, steps, penalties))
        print("Done training {} episodes!".format(train_episodes))

        with open("Taxi_Qtable.txt", "wb") as fp:  
            pickle.dump(self.q_table, fp)

    def load_qTable(self):
        with open("Taxi_Qtable.txt", "rb") as fp: 
            self.q_table = pickle.load(fp)

    def test(self, test_episodes):
        all_penalties = []
        all_steps = []
        all_rewards = []
        for episode in range(1,test_episodes+1):
            current_state = self.env.reset()
            done = False
            steps = 0
            penalties = 0
            rewards = 0
            while (steps < self.nsteps and not done):
                self.env.render()
                action = np.argmax(self.q_table[current_state])         
                new_state, reward, done, _ = self.env.step(action)
                current_state = new_state
                rewards+=reward
                steps += 1
                if reward == -10:
                    penalties += 1
                clear_output(wait=True)

            all_penalties.append(penalties)
            all_steps.append(steps)
            all_rewards.append(rewards)

            if(self.test_print):
                print("EPISODE : ", episode)
                print("Penalty : ",penalties)
                print("Steps : ",steps)
                print("Reward : ", rewards)
                print("-------------------\n")

        print("Average amount of steps: {}".format(np.mean(all_steps)))
        print("Average amount of penalties: {}".format(np.mean(all_penalties)))
        print("Average amount of rewards: {}".format(np.mean(all_rewards)))
    
env = gym.make("Taxi-v3").env

learning_rate = 0.05
gamma = 0.65
epsilon = 1.0
agent = QAgent(env,learning_rate, gamma, epsilon)
agent.train(50000) #Training (Making a Q-Table)
agent.test_print = 1
agent.load_qTable()
agent.test(3)



    