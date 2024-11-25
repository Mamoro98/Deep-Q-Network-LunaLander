##  🚀 Deep Q-Network Lunar Lander Notebook 🌙

## Overview 🧠✨

This repository contains a Jupyter Notebook implementation of a **Deep Q-Network (DQN)** to solve the 🌌 **Lunar Lander** environment from OpenAI's Gym. The goal of this project is to apply **reinforcement learning (RL)** techniques to train an agent capable of safely landing a spaceship 🛸 on a designated pad 🟩 within the environment. 

---

## Features 💡

- **Deep Q-Learning:** 📈 Utilizes a neural network to approximate Q-values for state-action pairs and updates them iteratively using the **Bellman equation**.
- **Replay Buffer:** 💾 Implements experience replay to store and sample past experiences for stable training.
- **Epsilon-Greedy Policy:** 🎲 Balances exploration and exploitation using a decaying epsilon-greedy policy.
- **Double Q-Networks:** 🔗 Uses separate online and target networks to stabilize the learning process.
- **Training Loop:** 🔄 Incorporates a robust training loop with episodic evaluations and the ability to save rendered episodes as videos 🎥.

---




## Prerequisites 🛠️

To run the notebook, ensure you have the following:

- **Python 3.x** 🐍
- **Jupyter Notebook** 📓
- Libraries:
  - `gym` 🏋️‍♀️
  - `jax` ⚡
  - `jaxlib` ⚡
  - `dm-haiku` 🧱
  - `optax` 🧮
  - `numpy` 🔢
  - `matplotlib` 📊
  - `chex` ✅

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## Notebook Structure 📂

1. **Setup 🛠️:**
   - Installs and imports all required packages.
   - Initializes the **Lunar Lander** environment from OpenAI Gym.

2. **Q-Learning Theory 📚:**
   - Explains key concepts like the **Bellman equation**, **greedy action selection**, and **Q-value updates**.

3. **Model Implementation 🧑‍💻:**
   - Builds the neural network using `dm-haiku` for Q-function approximation.
   - Implements functions for computing **squared loss**, **Bellman targets**, and **Q-learning loss**.

4. **Replay Buffer 💾:**
   - Creates a replay buffer to store transitions (`state`, `action`, `reward`, `next_state`, `done`).

5. **Training Loop 🔄:**
   - Trains the agent using a **policy-decay mechanism** (`epsilon-greedy`) and periodic updates of the target network.

6. **Evaluation 🎯:**
   - Logs training performance and plots the **episode rewards** to visualize learning progress.

---

## Key Functions 🔑

### Action Selection 🎲
- **Greedy Action Selection:** Chooses the action with the highest Q-value.
- **Random Action Selection:** Selects a random action for exploration.
- **Epsilon-Greedy Policy:** Combines greedy and random selection based on an epsilon schedule.

### Training 🏋️‍♂️
- **Loss Function:** Calculates the Q-learning loss using **squared error** between predicted Q-values and Bellman targets.
- **Optimizer:** Updates network parameters using `optax.adam`.

---

## Running the Notebook ▶️

Open **`luna-lander.ipynb`** and run the cells sequentially.

---

## Results 🏆

- The agent starts with **random actions** 🎲 and gradually learns to control the spaceship 🚀.
- Training episodes demonstrate improvement in **landing accuracy** 🟢 as the model optimizes the Q-function.
- Final evaluations show the agent's ability to **land consistently** with high reward scores 🌟.
https://github.com/user-attachments/assets/1cb1f589-d32d-4756-816d-79e7fb917222

---

## Contributions 🤝

This project was created as part of the **African Institute for Mathematical Sciences (AIMS)** program on Reinforcement Learning (2024) 🏫, led by **Arnu Pretorius**, Staff Research Scientist at InstaDeep 🧪. It is adapted from the **Deep Learning Indaba 2022** materials 🎓.

---

## License 📜

This project is licensed under the **Apache License 2.0**. 📄
