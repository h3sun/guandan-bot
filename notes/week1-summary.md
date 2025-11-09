# Week 1 Summary – MDP & Monte Carlo Self-Play

## Key Concepts
- **MDP**: 掼蛋可建模为一个多智能体马尔可夫决策过程 (state, actions, transition, reward)。
- **Monte Carlo RL**: 不依赖环境模型，从完整回合回报估计价值。
- **Self-Play**: 智能体与自身/旧版本对弈积累经验，形成自进化策略。

## Demo Implemented
文件: `rl_core/mc_selfplay_demo.py`
- 简化了2人对弈环境
- 每局生成完整episode, 用Monte Carlo更新Q表
- 可运行示例输出平均胜率 (~0.5)

## Next Steps
- 将MiniCardEnv替换为`guandan_mcc`环境
- 定义掼蛋状态/动作编码
- 扩展至4人自博弈 + 神经网络策略近似
