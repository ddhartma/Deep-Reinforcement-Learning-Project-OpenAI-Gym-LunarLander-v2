[image1]: assets/lunarlander_env.png "image1"

# Deep Reinforcement Learning Project - OpenAi Gym LunarLander-v2

## Content
- [Introduction](#intro)
- [OpenAI Gym - LunarLander-v2 - environment](#openai_lunarlander)
- [Files in the Repo](#files)
- [Implementation - deep_q_network.ipynb](#impl_notebook)
- [Implementation - dqn_agent.py](#impl_agent)
- [Implementation - model.py](#impl_model)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

## OpenAI Gym - LunarLander-v2 - environment <a name="openai_lunarlander"></a>
- Link to [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/)
- [Source Code](https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py) on Github

    ![image1]

- Landing pad is always at coordinates (0,0). 
- Coordinates are the first two numbers in state vector. 
- Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. 
- If lander moves away from landing pad it loses reward back. 
- Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. 
- Each leg ground contact is +10. 
- Firing main engine is -0.3 points each frame. 
- Solved is 200 points. 
- Landing outside landing pad is possible. 
- Fuel is infinite, so an agent can learn to fly and then land on its first attempt. 


- **Four discrete actions available**: 
    - do nothing, 
    - fire left orientation engine, 
    - fire main engine, 
    - fire right orientation engine.

## Files in the repo <a name="files"></a>
The workspace contains three files:
- **deep_q_network.ipynb**: Main file to implement DQN, notebook.
- **dqn_agent.py**: The reinforcement learning agent is developed.
- **model.py**: The interact function tests how well the agent learns from interaction with the environment.


## Implementation - dqn_agent.py <a name="impl_notebook"></a>
- Open Python file ```deep_q_network.ipynb```
    ```
    import gym
    !pip3 install box2d
    import random
    import torch
    import numpy as np
    from collections import deque
    import matplotlib.pyplot as plt
    %matplotlib inline

    !python -m pip install pyvirtualdisplay
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    ```
    ### Instantiate the Environment
    ```
    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    ```
    ### Instantiate the Agent
    ```
    from dqn_agent import Agent

    agent = Agent(state_size=8, action_size=4, seed=0)

    # watch an untrained agent
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))

    for j in range(200):
        action = agent.act(state)
        print(action)
        img.set_data(env.render(mode='rgb_array')) 
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        state, reward, done, _ = env.step(action)
        print('action: ', action)
        print('state: ', state)
        print('reward: ', reward)
        print('done: ', done)
        if done:
            break 
            
    env.close()

    RESULT:
    action:  0
    state:  [ -4.45019722e-02  -2.85196956e-02   9.58208057e-07  -7.08713372e-09
    9.33805568e-05   2.98583473e-08   1.00000000e+00   1.00000000e+00]
    reward:  -100
    done:  True
    ```
    ```
    def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """ Deep Q-Learning.
    
            INPUTS: 
            ------------
                n_episodes - (int) maximum number of training episodes
                max_t - (int) maximum number of timesteps per episode
                eps_start - (float) starting value of epsilon, for epsilon-greedy action selection
                eps_end - (float) minimum value of epsilon
                eps_decay - (float) multiplicative factor (per episode) for decreasing epsilon
                
            OUTPUTS:
            ------------
                scores - (list) list of scores
        """
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
        return scores

    scores = dqn()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    ```
    ### Watch a Smart Agent
    ```
    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    for i in range(3):
        state = env.reset()
        img = plt.imshow(env.render(mode='rgb_array'))
        for j in range(200):
            action = agent.act(state)
            img.set_data(env.render(mode='rgb_array')) 
            plt.axis('off')
            display.display(plt.gcf())
            display.clear_output(wait=True)
            state, reward, done, _ = env.step(action)
            if done:
                break 
                
    env.close()
    ```


## Implementation - dqn_agent.py <a name="impl_agent"></a>
- Open Python file ```dqn_agent.py```
    ```
    import numpy as np
    import random
    from collections import namedtuple, deque

    from model import QNetwork

    import torch
    import torch.nn.functional as F
    import torch.optim as optim

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ```
    ```
    class Agent():
        """ Interacts with and learns from the environment."""

        def __init__(self, state_size, action_size, seed):
            """ Initialize an Agent object.
            
                INPUTS: 
                ------------
                    state_size - (int) dimension of each state
                    action_size - (int) dimension of each action
                    seed - (int) random seed
                
                OUTPUTS:
                ------------
                    no direct
            """
            
            self.state_size = state_size
            self.action_size = action_size
            self.seed = random.seed(seed)

            # Q-Network
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

            # Replay memory
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
            # Initialize time step (for updating every UPDATE_EVERY steps)
            self.t_step = 0
        
        def step(self, state, action, reward, next_state, done):
            """ Update the agent's knowledge, using the most recently sampled tuple.
            
                INPUTS: 
                ------------
                    state - (array_like) the previous state of the environment (8,)
                    action - (int) the agent's previous choice of action
                    reward - (float) last reward received
                    next_state - (array_like) the current state of the environment
                    done - (bool) whether the episode is complete (True or False)
                
                OUTPUTS:
                ------------
                    no direct
            """
            # Save experience in replay memory
            self.memory.add(state, action, reward, next_state, done)
            
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    #print('experiences')
                    #print(experiences)
                    self.learn(experiences, GAMMA)

        def act(self, state, eps=0.):
            """ Returns actions for given state as per current policy.
            
                INPUTS:
                ------------
                    state - (array_like) current state
                    eps - (float) epsilon, for epsilon-greedy action selection

                OUTPUTS:
                ------------
                    act_select - (int) next epsilon-greedy action selection
            """
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy action selection
            if random.random() > eps:
                act_select = np.argmax(action_values.cpu().data.numpy())
                return act_select
            else:
                act_select = random.choice(np.arange(self.action_size))
                return act_select

        def learn(self, experiences, gamma):
            """ Update value parameters using given batch of experience tuples.

                INPUTS:
                ------------
                    experiences - (Tuple[torch.Variable]) tuple of (s, a, r, s', done) tuples 
                    gamma - (float) discount factor

                OUTPUTS:
                ------------
            """
            states, actions, rewards, next_states, dones = experiences

            # Compute and minimize the loss

            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            print(Q_targets_next)
            
            # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            print(Q_targets)

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)
            print(Q_expected)

            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

        def soft_update(self, local_model, target_model, tau):
            """ Soft update model parameters.
                θ_target = τ*θ_local + (1 - τ)*θ_target

                INPUTS:
                ------------
                    local_model - (PyTorch model) weights will be copied from
                    target_model - (PyTorch model) weights will be copied to
                    tau - (float) interpolation parameter 
                    
                OUTPUTS:
                ------------
                    no direct
                    
            """
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    RESULT print(experiences):
    (
        tensor([[8x floats for state], x 64 for minibatch ]),
        tensor([[1x int for action], x 64 for minibatch]),
        tensor([[1x float for reward], x 64 for minibatch]),
        tensor([[8x floats for next_state], x 64 for minibatch ]),
        tensor([[1x int for done], x 64 for minibatch ])
    )

    RESULT print(Q_targets_next):
    tensor([[1x float for Q value], x 64 for minibatch])

    RESULT print(Q_targets):
    tensor([[1x float for Q value], x 64 for minibatch])

    RESULT print(Q_expected):
    tensor([[1x float for Q value], x 64 for minibatch])
    ```

    ```
    class ReplayBuffer:
        """ Fixed-size buffer to store experience tuples."""

        def __init__(self, action_size, buffer_size, batch_size, seed):
            """ Initialize a ReplayBuffer object.

            INPUTS:
            ------------
                action_size - (int) dimension of each action
                buffer_size - (int) maximum size of buffer
                batch_size - (int) size of each training batch
                seed - (int) random seed
                
            OUTPUTS:
            ------------
                no direct
            """
            self.action_size = action_size
            self.memory = deque(maxlen=buffer_size)  
            self.batch_size = batch_size
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
            self.seed = random.seed(seed)
        
        def add(self, state, action, reward, next_state, done):
            """ Add a new experience to memory.
                
                INPUTS:
                ------------
                    state - (array_like) the previous state of the environment
                    action - (int) the agent's previous choice of action
                    reward - (int) last reward received
                    next_state - (int) the current state of the environment
                    done - (bool) whether the episode is complete (True or False)

                OUTPUTS:
                ------------
                    no direct
            
            """
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
        
        def sample(self):
            """ Randomly sample a batch of experiences from memory.
            
                INPUTS:
                ------------
                    None
                
                OUTPUTS:
                ------------
                    states - (torch tensor) the previous states of the environment
                    actions - (torch tensor) the agent's previous choice of actions
                    rewards - (torch tensor) last rewards received
                    next_states - (torch tensor) the next states of the environment
                    dones - (torch tensor) bools, whether the episode is complete (True or False)
            
            """
            experiences = random.sample(self.memory, k=self.batch_size)

            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
    
            return (states, actions, rewards, next_states, dones)

        def __len__(self):
            """ Return the current size of internal memory.
            
                INPUTS:
                ------------
                    None
                    
                OUTPUTS:
                ------------
                    mem_size - (int) current size of internal memory
            
            """
            mem_size = len(self.memory)
            return mem_size
        ```



        RESULT (print(self.qnetwork_local)):
        QNetwork(
            (fc1): Linear(in_features=8, out_features=64, bias=True)
            (fc2): Linear(in_features=64, out_features=64, bias=True)
            (fc3): Linear(in_features=64, out_features=4, bias=True)
        )
    ```

## Implementation - model.py <a name="openai_lunarlander"></a>
- Open Python file ```model.py```
    ```
    import torch.nn as nn
    import torch.nn.functional as F

    class QNetwork(nn.Module):
        """Actor (Policy) Model."""

        def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
            """ Initialize parameters and build model.
                INPUTS:
                ------------
                    state_size (int): Dimension of each state
                    action_size (int): Dimension of each action
                    seed (int): Random seed
                    fc1_units (int): Number of nodes in first hidden layer
                    fc2_units (int): Number of nodes in second hidden layer
                    
                OUTPUTS:
                ------------
                    no direct
            """
            super(QNetwork, self).__init__()
            self.seed = torch.manual_seed(seed)
            "*** YOUR CODE HERE ***"
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)
            

        def forward(self, state):
            """ Build a network that maps state -> action values.
                
                INPUTS:
                ------------
                    state - (array-like) actual 
                    
                OUTPUTS:
                ------------
                    output - (array-like) action values for given state set
            """
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            output = self.fc3(x)
            return output
    ```
   

## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Sparkify-Project.git
```

- Change Directory
```
$ cd Sparkify-Project
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
pyspark = 2.4.3
```

- Check the environment installation via
```
$ conda env list
```
### To Start taxi-v2 Training
- Open your terminal
- Navigate to ```main.py```
- Type in terminal
    ```
    python main.py
    ```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)