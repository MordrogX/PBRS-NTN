# 🧠 Potential-Based Reward Shaping Using Neural Tensor Network (PBRS-NTN)

Official repository for the B.Tech project *"Potential-Based Reward Shaping Using Neural Tensor Network"* by **Mukesh Meena**, under the supervision of **Prof. Kapil Ahuja** at **IIT Indore**.

This project introduces a novel approach to **reward shaping** in **Reinforcement Learning (RL)** using **Neural Tensor Networks (NTN)** to improve learning efficiency in sparse reward settings.

---

## 📌 Overview

Traditional RL environments often suffer from sparse and binary reward signals. This project presents a **Potential-Based Reward Shaping (PBRS)** algorithm that uses **Neural Tensor Networks** to learn transition dependencies and provide a more informative reward structure.

We compare this approach against:
- Sparse reward baseline
- Graph Convolutional Network (GCN)-based reward shaping

---

## 🧰 Features

- ✅ Implementation of Potential-Based Reward Shaping (PBRS)
- 🧠 Integration with Neural Tensor Network (NTN) to model transition relations
- 📈 Performance comparison with GCN-based reward shaping
- 🏃‍♂️ PPO (Proximal Policy Optimization) agent training on Mujoco's Ant-v2 environment

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/MordrogX/PBRS-NTN.git
cd PBRS-NTN
```

### 2. Install Dependencies
We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Run Training

To train the agent with different reward shaping techniques:

```bash
# For baseline PPO with sparse rewards
python train.py --method baseline

# For GCN-based reward shaping
python train.py --method gcn

# For PBRS with NTN
python train.py --method pbrs_ntn
```

### 4. Visualize Results

```bash
python plot_results.py
```

---

## 🧪 Environment Setup

- **Simulator**: [Mujoco](https://mujoco.org/) (via [OpenAI Gym](https://www.gymlibrary.dev/))
- **Environment**: `Ant-v2`
- **RL Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: PyTorch

---

## 📊 Results

Comparative evaluation on:
- Learning curve: Reward vs. Time steps
- Frame rate (FPS)
- Convergence rate

<p align="center">
  <img src="assets/reward_comparison_ntn_vs_gcn.png" width="70%" />
</p>

---

## 🧠 Neural Tensor Network

The **NTN** models the relationship between current and potential future states. It helps refine the PBRS function by predicting the quality of transition states more accurately than conventional reward estimators.

> Reference: Socher et al., *Reasoning with Neural Tensor Networks for Knowledge Base Completion* (2013)

---

## 📁 Project Structure

```
PBRS-NTN/
├── configs/                # Hyperparameter and environment configs
├── models/
│   ├── pbrs.py             # PBRS algorithm
│   ├── ntn.py              # Neural Tensor Network implementation
│   └── gcn.py              # GCN-based reward shaping
├── train.py                # Main training script
├── plot_results.py         # Visualization utility
├── requirements.txt
└── report/                 # Project report PDF
```

---

## 📚 References

1. Mahadevan & Maggioni (2007) - Proto-value functions via Laplacian
2. Klissarov et al. - Reward shaping using Graph Convolutional Networks
3. Socher et al. - Neural Tensor Networks for reasoning
4. Brys et al. - Multi-objectivization of reinforcement learning

---

## 👨‍💻 Author

**Mukesh Meena**  
B.Tech, Computer Science & Engineering  
[Indian Institute of Technology Indore](https://www.iiti.ac.in)  
🔗 [LinkedIn](https://www.linkedin.com/in/mukesh-meena/)  
📧 [Email](mailto:mukeshmeena.cse@gmail.com)

---

## 📜 License

This project is intended for **academic and research use** only.  
© 2022 Mukesh Meena, IIT Indore.
