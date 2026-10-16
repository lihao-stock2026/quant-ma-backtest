"""
项目名称：滑动窗口滚动回测（Walk‑Forward Analysis）—— 样本外测试
功能：避免"用未来数据找最优参数"的过拟合问题
     每一段历史数据用前半段（训练期）找最优均线参数，
     再用后半段（测试期）验证该参数的真实表现。
> 风险提示：滚动回测仍属于历史回测，不代表未来实盘收益。
"""
import argparse
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from backtest_utils import load_stock_data
warnings.filterwarnings("ignore")


def backtest_single(df, short_window, long_window):
    """单组参数回测，返回净值序列和夏普比率"""
    d = df.copy()
    d["ma_s"] = d["close"].rolling(short_window).mean()
    d["ma_l"] = d["close"].rolling(long_window).mean()
    d = d.dropna(subset=["ma_s", "ma_l"]).reset_index(drop=True)

    d["gold"] = (d["ma_s"] > d["ma_l"]) & (d["ma_s"].shift(1) <= d["ma_l"].shift(1))
    d["dead"] = (d["ma_s"] < d["ma_l"]) & (d["ma_s"].shift(1) >= d["ma_l"].shift(1))

    d["signal"] = 0
    pos = 0
    for i in range(len(d)):
        if d.loc[i, "gold"]:
            pos = 1
        if d.loc[i, "dead"]:
            pos = 0
        d.loc[i, "signal"] = pos

    d["ret"] = d["close"].pct_change()
    d = d.dropna(subset=["ret"])
    d["strat_ret"] = d["ret"] * d["signal"].shift(1)
    cum = (1 + d["strat_ret"]).cumprod()

    if len(d) < 2:
        return None, None, None

    std = d["strat_ret"].std()
    sharpe = np.nan if std == 0 else (d["strat_ret"] - 0.02/252).mean() / std * np.sqrt(252)
    return cum, sharpe, d


def grid_search_best(train_df, short_list, long_list):
    """在训练集上做网格搜索，返回夏普最高的参数组合"""
    best_sharpe = -np.inf
    best_params = None
    for s in short_list:
        for l in long_list:
            if s >= l:
                continue
            _, sharpe, _ = backtest_single(train_df, s, l)
            if sharpe is not None and not np.isnan(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (s, l)
    return best_params, best_sharpe


def walk_forward(df_raw, short_list, long_list, train_days=120, test_days=60, step_days=60):
    """滑动窗口滚动回测"""
    df = df_raw.copy().reset_index(drop=True)
    n = len(df)

    results = []
    all_test_dates = []
    all_test_net = []
    capital = 1.0

    start = 0
    fold = 1
    while start + train_days + test_days <= n:
        train_df = df.iloc[start : start + train_days].copy()
        test_df = df.iloc[start + train_days - 1 : start + train_days + test_days].copy()

        best_params, train_sharpe = grid_search_best(train_df, short_list, long_list)
        if best_params is None:
            start += step_days
            fold += 1
            continue

        s, l = best_params
        test_cum, test_sharpe, test_detail = backtest_single(test_df, s, l)
        if test_cum is None or len(test_detail) == 0:
            start += step_days
            fold += 1
            continue

        test_cum_cont = capital * test_cum
        all_test_net.extend(test_cum_cont.values.tolist())
        all_test_dates.extend(test_detail["trade_date"].values.tolist())
        capital = test_cum_cont.iloc[-1]

        test_total_ret = test_cum.iloc[-1] - 1
        bench_ret = test_detail["close"].iloc[-1] / test_detail["close"].iloc[0] - 1

        results.append({
            "轮次": fold,
            "训练区间": f"{train_df['trade_date'].iloc[0].date()}~{train_df['trade_date'].iloc[-1].date()}",
            "测试区间": f"{test_detail['trade_date'].iloc[0].date()}~{test_detail['trade_date'].iloc[-1].date()}",
            "最优短均线": s,
            "最优长均线": l,
            "训练期夏普": round(train_sharpe, 3),
            "测试期策略收益": f"{test_total_ret:.2%}",
            "测试期基准收益": f"{bench_ret:.2%}",
            "测试期夏普": round(test_sharpe, 3) if not np.isnan(test_sharpe) else np.nan
        })

        start += step_days
        fold += 1

    if len(all_test_net) > 0:
        oos_cum = pd.Series(all_test_net)
        oos_dates = pd.to_datetime(all_test_dates)
    else:
        oos_cum = pd.Series(dtype=float)
        oos_dates = pd.Series(dtype=object)

    return pd.DataFrame(results), oos_dates, oos_cum


def main():
    parser = argparse.ArgumentParser(description="滑动窗口滚动样本外回测")
    parser.add_argument("--code", type=str, help="股票代码，例如600519.SH")
    parser.add_argument("--start", type=str, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, help="结束日期 YYYYMMDD")
    args = parser.parse_args()

    ts_code = args.code if args.code else input("请输入股票代码（例：600519.SH）：").strip()
    start_date = args.start if args.start else input("回测开始日期(YYYYMMDD)，回车默认20220101：").strip() or "20220101"
    end_date = args.end if args.end else input("回测结束日期(YYYYMMDD)，回车默认20241231：").strip() or "20241231"

    df_raw = load_stock_data(ts_code, start_date, end_date)
    if df_raw is None or df_raw.empty:
        print("❌ 行情数据获取失败，程序退出")
        return

    short_list = [5, 7, 10, 12, 15]
    long_list = [20, 30, 40, 60]

    print(f"\n开始滚动回测，训练窗口120交易日 / 测试窗口60交易日 / 滑动60交易日...")
    print("-" * 60)

    fold_df, oos_dates, oos_cum = walk_forward(
        df_raw, short_list, long_list,
        train_days=120, test_days=60, step_days=60
    )

    print("\n" + "=" * 60)
    print(f"{ts_code} 滚动回测各轮次结果")
    print("=" * 60)
    print(fold_df.to_string(index=False))

    fold_df.to_csv("walk_forward_result.csv", index=False, encoding="utf-8-sig")
    if len(oos_cum) > 0:
        oos_df_out = pd.DataFrame({"oos_net": oos_cum.values, "oos_date": oos_dates.values})
        oos_df_out.to_csv("oos_sample_out_net.csv", index=False, encoding="utf-8-sig")
        print("✅ 样本外净值已保存：oos_sample_out_net.csv")

    print("\n✅ 各轮次结果已保存：walk_forward_result.csv")

    print("\n" + "=" * 60)
    print("对比：全程最优参数（样本内）vs 滚动窗口参数（样本外）")
    print("=" * 60)

    best_params_full, train_sharpe_full = grid_search_best(df_raw, short_list, long_list)
    if best_params_full is None:
        print("❌ 全程网格寻优未找到有效参数，数据不足！")
        return

    full_cum, full_sharpe, full_detail = backtest_single(df_raw, *best_params_full)

    print(f"全程最优参数：MA{best_params_full[0]}/MA{best_params_full[1]}")
    print(f"  样本内最终净值：{full_cum.iloc[-1]:.3f}，夏普：{full_sharpe:.3f}")
    if len(oos_cum) > 0:
        print(f"  样本外（滚动拼接）最终净值：{oos_cum.iloc[-1]:.3f}")
        print(f"  → 如果样本外净值明显低于样本内，说明参数过拟合严重")

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    axes[0].plot(full_detail["trade_date"], full_cum.values, label=f"样本内全程最优 MA{best_params_full[0]}/MA{best_params_full[1]}", color="red")
    axes[0].set_title("样本内回测（用全部历史数据找参数 → 过拟合风险高）")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if len(oos_cum) > 0:
        axes[1].plot(range(len(oos_cum)), oos_cum.values, label="样本外滚动净值（Walk‑Forward，X为交易日序号）", color="green")
    axes[1].set_title("样本外回测（每段用训练期找参数 → 测试期验证，更接近真实）")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    img_name = f"{ts_code.replace('.','_')}_walkforward.png"
    plt.savefig(img_name, dpi=300)
    plt.show()
    print(f"\n✅ 滚动回测对比图已保存：{img_name}")

    print("\n" + "=" * 60)
    print("📚 知识点：为什么要做样本外测试？")
    print("=" * 60)
    print("1. 样本内回测：用全部历史数据找最优参数，天然带'未来函数'")
    print("2. 样本外测试：只允许用过去的数据找参数，在未来数据上验证")
    print("3. 如果样本外收益大幅下降 → 说明参数是'拟合'出来的，实盘会亏")
    print("4. 这是量化面试高频考点：如何避免过拟合？")


if __name__ == "__main__":
    main()