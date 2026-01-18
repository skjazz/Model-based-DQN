# Gated Rainbow DQN with World Model

A reinforcement learning agent combining gated linear networks with Rainbow DQN enhancements and a world model for improved sample efficiency and performance.

## 🚀 Features

- **Gated Linear Network**: Adaptive feature selection through gating mechanisms
- **Rainbow DQN Enhancements**:
  - Noisy Networks for exploration
  - Prioritized Experience Replay (PER)
  - N-step returns
  - Dueling architecture
  - Double DQN
- **World Model Integration**: Dream-like experience generation
- **Dynamic Environments**: Custom NumberGame environment with varying difficulty
- **Memory Efficient**: Impression buffers for sequence handling

## 📋 Requirements

torch>=2.0.0
numpy>=1.24.0
gymnasium>=0.29.0
tensordict>=0.2.0
torchrl>=0.2.0


## 🏗️ Architecture

### Core Components

1. **GatedDQN**: Main agent network with:
   - Gated linear layers for adaptive feature selection
   - Noisy linear layers for exploration
   - Dueling streams for value/advantage separation
   - Sequence-aware gate network

2. **World Model**: Predictive model for:
   - State transition prediction
   - Dream experience generation
   - Model-based planning

3. **Agent**:
   - PER with dynamic α, β parameters
   - N-step returns
   - Mixed training (real + dream experiences)

### Environment: NumberGame

A custom Gymnasium environment featuring:
- Health/Mana resource management
- Multiple action strategies (healing, attacking, drawing)
- Dynamic enemy damage patterns
- Progressive difficulty scaling
- Complex reward shaping

## 🚦 Quick Start


### Training


python train.py


### Key Training Features

- **Automatic difficulty scaling**: Environment parameters adjust over time
- **Dream generation**: World model creates synthetic experiences
- **Dynamic exploration**: Epsilon decay with noisy networks
- **Learning rate scheduling**: Adaptive LR based on performance

## 📊 Hyperparameters


EPISODES = 400000
LR = 0.0001
BATCH_SIZE = 256
GAMMA = 0.999
MEMORY_SIZE = 100000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
SEQ_LENGTH = 4  # For sequence modeling

## 🎮 Action Space

| Action | Strategy | Description |
|--------|----------|-------------|
| 0 | [1, 1, 2] | Light heal ×2 + Heavy heal |
| 1 | [1, 2] | Light heal + Heavy heal |
| 2 | [3] | Delayed heal |
| 3 | [4] | Draw (special action) |

## 🏆 Performance Tracking

The agent tracks:
- Average reward (100-episode window)
- Training loss
- World model prediction loss
- Dream usage statistics
- Gate activation patterns
- Exploration rate decay

## 📈 Results

The agent achieves progressive improvement through:
1. **Early Phase**: Random exploration, basic policy learning
2. **Mid Phase**: PER optimization, world model integration
3. **Late Phase**: Dream-enhanced training, refined gating

## 🔧 Technical Details

### Gating Mechanism
- Processes sequence history (SEQ_LENGTH=4)
- Produces modulation factors ∈ [0.7, 1.0]
- Regularized to prevent collapse

### Dream Generation
- Starts at episode 800
- 30% of training batch from dreams
- 5 dreams generated per step
- Each dream 50 steps long

### Prioritized Experience Replay
- Dynamic α: 0.5 → 0.7 → 0.4
- Dynamic β: 0.4 → 1.0
- Importance sampling weights


## 📁 Project Structure


project/
├── train.py              # Main training script
├── requirements.txt      # Dependencies
├── LICENSE              # MIT License
└── README.md           # This file

## 📝 Key Implementation Notes

1. **Device Management**: Automatic CUDA/CPU detection
2. **Memory Management**: Efficient buffer handling for long training
3. **Reproducibility**: Seeded randomness where appropriate
4. **Logging**: Progressive logging with detailed episodes

## 🧪 Experimental Features

- **Hybrid Training**: Real + synthetic experiences
- **Adaptive Gating**: Context-aware feature selection
- **Progressive Difficulty**: Self-adjusting environment
- **Dynamic Exploration**: Multiple exploration strategies

## 🔮 Future Work

1. **Architectural Improvements**:
   - Transformer-based gating
   - Multi-head attention mechanisms
   - Hierarchical world models

2. **Algorithmic Enhancements**:
   - Distributional RL (C51/QR-DQN)
   - Curiosity-driven exploration
   - Meta-learning for fast adaptation

3. **Infrastructure**:
   - Distributed training
   - Web-based visualization
   - Experiment tracking (Weights & Biases)

## 📚 References

- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
- [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
- [World Models](https://arxiv.org/abs/1803.10122)
- [Gated Linear Networks](https://arxiv.org/abs/1910.01526)

## 👥 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Gymnasium team for the RL environment interface
- DeepMind for foundational Rainbow DQN research

---

*For questions, issues, or contributions, please open an issue or pull request on GitHub.*
