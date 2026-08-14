# Intelligent Supply Chain Management System

A multi-echelon supply chain simulator with six AI components integrated into a single
decision-making pipeline: a Gymnasium environment, an information-sharing network, a transformer
demand forecaster, a fuzzy controller, a multi-objective evolutionary optimiser, and an actor-critic
reinforcement learning policy.

Built as an undergraduate honours project (BEng Robotics, Autonomous & Interactive Systems,
Heriot-Watt University, 2024–25).

## The problem

Supply chains distort information as it moves upstream. A small change in customer demand amplifies
at each echelon until the factory sees swings several times larger than anything a customer did —
the **bullwhip effect**. The usual responses trade against each other: hold more inventory and cost
rises, hold less and service level falls, and neither addresses the distortion itself.

This project asks whether combining several techniques that each address part of the problem does
better than any of them alone, and provides an environment to test that.

The engineering interest is less in any single component than in **making six of them cooperate**.
Each works in isolation; getting a learned forecaster, a rule-based controller, an evolutionary
optimiser and an RL policy to inform one decision without fighting each other is the actual problem.

## Architecture

![System architecture](docs/system_architecture.png)

Information flows through the components in order, each one enriching what the next receives:

| Component | Role |
|---|---|
| **Environment** | Multi-echelon supply chain on a Gymnasium interface — inventory dynamics, demand patterns, cost and service-level metrics |
| **Information Sharing Network** | Propagates state between echelons to reduce information distortion before decisions are made |
| **Transformer forecaster** | Sequence model predicting future demand, with probabilistic output so uncertainty is available downstream |
| **Fuzzy controller** | Encodes domain heuristics as rules, handling the imprecision that point forecasts hide |
| **MOEA/D** | Optimises the competing objectives — total cost, service level, bullwhip — producing Pareto-optimal parameter sets rather than one compromise |
| **Actor-critic policy** | Takes the enriched state, forecast, and recommendations, and makes the ordering decision |

Detailed write-ups for each component are in [`docs/`](docs/), including the integration notes for
the transformer/ISN boundary and the fuzzy controller revisions.

## Quickstart

```bash
git clone https://github.com/AMMTAS1010/SupplyChain.git
cd SupplyChain

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Conda users can use `environment.yml` instead.

Run the integrated system:

```python
from main import SupplyChainSystem, SystemConfig

system = SupplyChainSystem(SystemConfig())
system.train(n_episodes=1000)
```

With a custom configuration:

```python
config = SystemConfig(
    n_echelons=4,
    demand_history_len=24,
    episode_length=100,
    hidden_size=64,
    n_heads=4,
    n_layers=3,
    dropout=0.1,
    learning_rate=1e-4,
)
system = SupplyChainSystem(config)
```

Individual components can be trained on their own via [`scripts/`](scripts/):

```bash
python scripts/train_transformer.py
python scripts/train_actor_critic.py
```

A pre-trained forecaster is included as `transformer_model.pt`.

## Tests

```bash
pytest -v
```

Environment tests cover the reset and step contract, inventory and backlog accounting, cost
computation and the observation space.

## Project structure

```
├── config/                     # Configuration dataclasses per component
│   ├── information_sharing_config.py
│   └── transformer_config.py
├── docs/                       # Architecture and per-component write-ups
│   ├── architecture.md
│   ├── system_architecture.png
│   ├── information_sharing_network.md
│   ├── transformer_isn_integration.md
│   ├── fuzzy_controller_updates.md
│   └── research_documentation.md
├── results/                    # Training-run telemetry (JSON)
├── scripts/                    # Standalone component training
├── src/
│   ├── environment/            # Gymnasium supply chain simulator
│   ├── models/
│   │   ├── information_sharing/
│   │   ├── transformer/
│   │   ├── fuzzy/
│   │   ├── moea/
│   │   └── actor_critic/
│   └── utils/                  # Metrics, time features
├── tests/
├── main.py                     # Integrated system entry point
├── requirements.txt
└── environment.yml
```

`results/` contains JSON telemetry captured during training runs — episode reward, service level,
bullwhip measure and per-network losses. They are run records, not benchmark results.

## Theoretical basis

Each component implements a documented approach rather than an ad-hoc design.

**Bullwhip effect and information sharing**
- Lee, H. L., Padmanabhan, V., & Whang, S. (1997). Information distortion in a supply chain: the bullwhip effect. *Management Science*, 43(4), 546–558.
- Chen, F., Drezner, Z., Ryan, J. K., & Simchi-Levi, D. (2000). Quantifying the bullwhip effect in a simple supply chain. *Management Science*, 46(3), 436–443.

**Demand forecasting**
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Salinas, D., et al. (2020). DeepAR: probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181–1191.

**Fuzzy control**
- Petrovic, D., Roy, R., & Petrovic, R. (1999). Supply chain modelling using fuzzy sets. *International Journal of Production Economics*, 59(1–3), 443–453.
- Zadeh, L. A. (1996). Fuzzy logic = computing with words. *IEEE Transactions on Fuzzy Systems*, 4(2), 103–111.

**Multi-objective optimisation**
- Zhang, Q., & Li, H. (2007). MOEA/D: a multiobjective evolutionary algorithm based on decomposition. *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.
- Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting. *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.

**Reinforcement learning**
- Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-critic algorithms. *NeurIPS*.
- Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning. *ICLR*.

## Licence

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{alshaqra_supply_chain_2025,
  title  = {Intelligent Supply Chain Management System},
  author = {Alshaqra, Abdallah},
  year   = {2025},
  url    = {https://github.com/AMMTAS1010/SupplyChain}
}
```
