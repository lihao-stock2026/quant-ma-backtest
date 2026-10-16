# 均线量化回测项目
> 课程Python量化作业 | 双均线策略、止盈止损风控、Walk‑Forward滚动窗口样本外验证
> ⚠️ 风险提示：本项目全部结果均为历史回测，存在过拟合风险，**不代表未来实盘收益，不构成任何投资建议**

## 📁项目文件说明
| 文件 | 作用 |
|---|---|
| `backtest_utils.py` | 公共工具模块（**无需直接运行**），封装Tushare行情读取缓存、回测指标计算、异常捕获，消除代码冗余 |
| `ma_strategy_stoploss.py` | 任务4：纯均线策略 vs 均线+止盈止损风控对比回测 |
| `walkforward_backtest.py` | 任务5：滑动窗口Walk‑Forward样本外回测，抑制参数过拟合 |
| `report_generator.py` | 任务6：一键生成HTML完整回测报告，包含图表、指标、逐笔交易记录 |
| `data/` | 自动生成目录，存放Tushare股票K线缓存csv，减少重复网络请求 |

## 📦环境依赖安装
```bash
pip install tushare pandas numpy matplotlib