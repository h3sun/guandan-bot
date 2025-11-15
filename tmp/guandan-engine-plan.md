# Guandan Engine Plan

## 1. Current State Overview
- Repo only has `rl_core/mc_selfplay_demo.py`, a stub Monte Carlo self-play demo without real Guandan logic.
- No card, pattern, or match management modules exist; no level/tribute mechanics are implemented.
- Tests, data models, and environment interfaces for Guandan are absent.

## 2. Target Final State Overview
- Introduce a deterministic Guandan rules engine that validates plays, compares patterns, drives trick flow, and tracks finish order.
- Support tribute/anti-tribute calculations plus level progression between hands inside a reusable match state object.
- Provide pure functional APIs so RL/self-play code can query legal moves, apply actions, and reset hands cleanly.

## 3. Files To Change
1. `rl_core/guandan_engine.py` (new): implement all pure functions for card modeling, pattern classification/comparison, move generation, tribute, and level updates.
2. `rl_core/mc_selfplay_demo.py`: replace placeholder environment hooks with calls into the new engine to enable realistic rollouts.
3. `README`: expand usage notes to mention the Guandan engine module and how to run demos.

## 4. Task Checklist
- [ ] Understand rule spec details needed for data structures and ordering.
  - [ ] Define card, pattern, and state data models with strict typing.
- [ ] Implement pattern classification with trump wildcard handling.
  - [ ] Enforce max lengths for straights,连对,钢板, same-type comparisons.
- [ ] Implement pattern comparison hierarchy (bomb tiers, four jokers).
- [ ] Build turn/trick flow helpers: legal move generation, apply move, finish order detection.
  - [ ] Record finished order and detect hand termination.
- [ ] Implement tribute + anti-tribute logic and who leads next hand.
- [ ] Implement level progression updates and match completion condition (pass A).
- [ ] Integrate engine into Monte Carlo demo + update README instructions.

## 5. Future Ideas (not in scope)
- Add Gym-style RL interface plus caching for combinational move enumeration.
- Implement tournament scoring tables and persistence for multi-match tracking.
