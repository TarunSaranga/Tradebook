from tensorforce.agents import DQNAgent
from TradeEnvironment import TradeEnvironment
import os
import time as t
import numpy as np
import matplotlib.pyplot as plt


###############################_Directories_###################################

saver_dir = './model/bot'+str(int(t.time()))
if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)

summarizer_dir = './summerizer/DQN/bot/'+str(int(t.time()))
if not os.path.exists(summarizer_dir):
    os.makedirs(summarizer_dir)

recorder_dir = './recorder/DQN/bot'+str(int(t.time()))
if not os.path.exists(recorder_dir):
    os.makedirs(recorder_dir)

###############################################################################
"""
Change values here to run
"""
num_episodes = 1
learning_rate_spec = 0.01
exp_decay_rate = 0.5
discount_rate = 0.9
###############################################################################
if(num_episodes == 1):
    notebook = TradeEnvironment(viz=True)
else:
    notebook = TradeEnvironment()

layer_spec = [
              dict(type='internal_lstm',length=4,size=128,activation='relu'),
              dict(type='cov2d',size=128,activation='relu'),
              dict(type='dense',size=128,activation='relu'),
             ]

network_spec = dict(
                    type="layered",
                    layers=layer_spec
                    )

exploration_spec = dict(
        type='decaying',
        dtype="float",
        unit="episodes" if(num_episodes>1) else "timesteps",
        decay="exponential",
        initial_value=1.0,
        decay_steps=num_episodes if(num_episodes>1) else int(notebook._max_timesteps),
        decay_rate=exp_decay_rate
        )



agent = DQNAgent(states = notebook.states(),
                 actions = notebook.actions(),
                 max_episode_timesteps = notebook.max_episode_timesteps(),
                 memory = notebook.max_episode_timesteps() +100,
                 learning_rate=learning_rate_spec,
                 network = network_spec,
                 exploration = exploration_spec,
                 update_frequency = 32,
                 discount=discount_rate,
                 saver = dict(directory = saver_dir),
                 summarizer=dict(directory = summarizer_dir,labels=["graph","episode-reward"]),
                 recorder =dict(directory= recorder_dir)
                 )

agent.initialize()

#%%
###Train Agent
step_plot = 0
step_print = 0
episode_plot = 1
episode_print = 1
if(num_episodes==1):
    step_plot=1
    step_print=1
    episode_plot = 0
    episode_print = 0
elif(num_episodes>1):
    step_plot=0
    step_print=0
    episode_plot=1
    episode_print=1
avgreward_per_episode = []
profits_per_episode = []
avg_ep_reward = 0
for i in range(num_episodes):
    agent.reset()
    states = notebook.reset()
    terminal = False
    step_reward = []
    print("Episode: "+str(i))
    while not terminal:
        action = agent.act(states=states)
        position = notebook.position
        states,terminal,reward = notebook.execute(actions=action)
        agent.observe(reward=reward,terminal=terminal)
        if(step_print):
            print("TS: "+str(notebook.time_step)+" action: "+str(action)
            +" position: "+str(position)+" reward: "+str(reward)
            +" profit: "+str(notebook.profit)+" curr_price: "+str(notebook.curr_price)
            +" curr_cash: "+str(notebook.curr_cash))
        step_reward.append(reward)
    final_profit = notebook.curr_cash - notebook.starting_cash
    if(episode_print):
        print(" FinalCash: "+str(notebook.curr_cash)+
        " Profit: "+str(notebook.curr_cash-notebook.starting_cash)+" MeanReward: "+str(np.array(step_reward).mean()))
    profits_per_episode.append(final_profit)
    if(step_plot):
        plt.figure(figsize=(20,12))
        plt.title("Reward per Step | Profit= "+str(final_profit)+" Tot Reward= "+str(sum(step_reward)))
        plt.xlabel("Step #")
        plt.ylabel("Reward")
        plt.plot(step_reward)
    avg_ep_reward = np.array(step_reward).mean()
    avgreward_per_episode.append(avg_ep_reward)

if(episode_plot):
    plt.figure(figsize=(20,12))
    plt.title("Avg Reward per Episode | Avg RewardperEpisode= "+str(np.array(avgreward_per_episode).mean()))
    plt.xlabel("Episode #")
    plt.ylabel("Avg Reward")
    plt.plot(avgreward_per_episode)
    plt.figure(figsize=(20,12))
    plt.title("Profits per Episode | Avg Profit= "+str(np.array(profits_per_episode).mean()))
    plt.xlabel("Episode #")
    plt.ylabel("Profits")
    plt.plot(profits_per_episode)

agent.save(saver_dir)

#%%
'''
This part of the code is used to test the agent
'''
print("########################################################################")
print("Testing Plots")
print("########################################################################")

##%%
####Test Agent
step_plot = 0
step_print = 0
episode_plot = 1
episode_print = 1
if(num_episodes==1):
    step_plot=1
    step_print=1
    episode_plot = 0
    episode_print = 0
elif(num_episodes>1):
    step_plot=0
    step_print=0
    episode_plot=1
    episode_print=1
avgreward_per_episode = []
profits_per_episode = []
avg_ep_reward = 0
for i in range(num_episodes):
    agent.reset()
    states = notebook.reset()
    terminal = False
    step_reward = []
    print("Episode: "+str(i))
    while not terminal:
        action = agent.act(states=states, deterministic=True)
        position = notebook.position
        states,terminal,reward = notebook.execute(actions=action)
        if(step_print):
            print("TS: "+str(notebook.time_step)+" action: "+str(action)
            +" position: "+str(position)+" reward: "+str(reward)
            +" profit: "+str(notebook.profit)+" curr_price: "+str(notebook.curr_price)
            +" curr_cash: "+str(notebook.curr_cash))
        step_reward.append(reward)
    final_profit = notebook.curr_cash - notebook.starting_cash
    if(episode_print):
        print(" FinalCash: "+str(notebook.curr_cash)+
        " Profit: "+str(notebook.curr_cash-notebook.starting_cash)+" MeanReward: "+str(np.array(step_reward).mean()))
    profits_per_episode.append(final_profit)
    if(step_plot):
        plt.figure(figsize=(20,12))
        plt.title("Reward per Step | Profit= "+str(final_profit)+" Tot Reward= "+str(sum(step_reward)))
        plt.xlabel("Step #")
        plt.ylabel("Reward")
        plt.plot(step_reward)
    avg_ep_reward = np.array(step_reward).mean()
    avgreward_per_episode.append(avg_ep_reward)

if(episode_plot):
    plt.figure(figsize=(20,12))
    plt.title("Avg Reward per Episode | Avg RewardperEpisode= "+str(np.array(avgreward_per_episode).mean()))
    plt.xlabel("Episode #")
    plt.ylabel("Avg Reward")
    plt.plot(avgreward_per_episode)
    plt.figure(figsize=(20,12))
    plt.title("Profits per Episode | Avg Profit= "+str(np.array(profits_per_episode).mean()))
    plt.xlabel("Episode #")
    plt.ylabel("Profits")
    plt.plot(profits_per_episode)