"""
项目名称：均线策略 + 止盈止损风控对比
功能：对比【纯均线金叉死叉】vs【均线+止盈止损风控】两种策略的回测效果
> 风险提示：历史回测结果存在过拟合风险，不代表未来实盘收益，仅用于学习演示。
"""
import argparse
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from backtest_utils import load_stock_data, calc_metrics
warnings.filterwarnings("ignore")


def calc_base_strategy(df_raw, short_window=5, long_window=20):
    """纯均线金叉死叉策略（无风控）"""
    df = df_raw.copy()
    df["ma_short"] = df["close"].rolling(window=short_window).mean()
    df["ma_long"] = df["close"].rolling(window=long_window).mean()
    if len(df) == 0:
        print("⚠️ calc_base_strategy：原始数据为空！")
        return None, None, None, None
    df = df.dropna(subset=["ma_short", "ma_long"]).reset_index(drop=True)
    if len(df) < 2:
        print("⚠️ calc_base_strategy：均线计算后有效数据不足！")
        return None, None, None, None

    df["gold_cross"] = (df["ma_short"] > df["ma_long"]) & (df["ma_short"].shift(1) <= df["ma_long"].shift(1))
    df["dead_cross"] = (df["ma_short"] < df["ma_long"]) & (df["ma_short"].shift(1) >= df["ma_long"].shift(1))

    df["signal"] = 0
    position = 0
    buy_index = None
    trade_records = []
    for i in range(len(df)):
        if df.loc[i, "gold_cross"] and position == 0:
            position = 1
            buy_index = i
        if df.loc[i, "dead_cross"] and position == 1:
            position = 0
            trade_records.append({
                "买入日期": df.loc[buy_index, "trade_date"],
                "买入价": df.loc[buy_index, "close"],
                "卖出日期": df.loc[i, "trade_date"],
                "卖出价": df.loc[i, "close"]
            })
            buy_index = None
        df.loc[i, "signal"] = position
    # 期末强制平仓
    if position == 1 and buy_index is not None:
        trade_records.append({
            "买入日期": df.loc[buy_index, "trade_date"],
            "买入价": df.loc[buy_index, "close"],
            "卖出日期": df.iloc[-1]["trade_date"],
            "卖出价": df.iloc[-1]["close"]
        })

    df["return"] = df["close"].pct_change()
    df = df.dropna(subset=["return"])
    if len(df) < 2:
        print("⚠️ calc_base_strategy：收益率计算后数据不足！")
        return None, None, None, None
    df["strategy_return"] = df["return"] * df["signal"].shift(1)
    strategy_cum = (1 + df["strategy_return"]).cumprod()

    metrics_dict = calc_metrics(df, df["strategy_return"])
    metrics = {
        "策略类型": "纯均线（无风控）",
        "最终净值": metrics_dict["最终净值"],
        "年化收益率": metrics_dict["年化收益率"],
        "最大回撤": metrics_dict["最大回撤"],
        "夏普比率": metrics_dict["夏普比率"],
        "交易次数": len(trade_records)
    }
    return df, strategy_cum, pd.DataFrame(trade_records), metrics


def calc_stoploss_strategy(df_raw, short_window=5, long_window=20,
                           stop_loss_pct=0.08, take_profit_pct=0.15):
    """均线策略 + 止盈止损风控"""
    df = df_raw.copy()
    df["ma_short"] = df["close"].rolling(window=short_window).mean()
    df["ma_long"] = df["close"].rolling(window=long_window).mean()
    if len(df) == 0:
        print("⚠️ calc_stoploss_strategy：原始数据为空！")
        return None, None, None, None
    df = df.dropna(subset=["ma_short", "ma_long"]).reset_index(drop=True)
    if len(df) < 2:
        print("⚠️ calc_stoploss_strategy：均线计算后有效数据不足！")
        return None, None, None, None

    df["gold_cross"] = (df["ma_short"] > df["ma_long"]) & (df["ma_short"].shift(1) <= df["ma_long"].shift(1))
    df["dead_cross"] = (df["ma_short"] < df["ma_long"]) & (df["ma_short"].shift(1) >= df["ma_long"].shift(1))

    df["signal"] = 0
    df["sell_reason"] = ""
    position = 0
    buy_price = 0.0
    trade_records = []
    for i in range(len(df)):
        price = df.loc[i, "close"]
        if position == 0:
            if df.loc[i, "gold_cross"]:
                position = 1
                buy_price = price
                df.loc[i, "signal"] = 1
                trade_records.append({
                    "买入日期": df.loc[i, "trade_date"],
                    "买入价": price,
                    "卖出日期": None,
                    "卖出价": None,
                    "卖出原因": None,
                    "单笔收益": None
                })
        else:
            profit_pct = (price - buy_price) / buy_price
            sell_reason = None
            if profit_pct <= -stop_loss_pct:
                sell_reason = "止损"
            elif profit_pct >= take_profit_pct:
                sell_reason = "止盈"
            elif df.loc[i, "dead_cross"]:
                sell_reason = "死叉平仓"

            if sell_reason is not None:
                position = 0
                df.loc[i, "signal"] = 0
                df.loc[i, "sell_reason"] = sell_reason
                if trade_records and trade_records[-1]["卖出日期"] is None:
                    trade_records[-1]["卖出日期"] = df.loc[i, "trade_date"]
                    trade_records[-1]["卖出价"] = price
                    trade_records[-1]["卖出原因"] = sell_reason
                    trade_records[-1]["单笔收益"] = f"{profit_pct:.2%}"
            else:
                df.loc[i, "signal"] = 1
    # 期末强制平仓
    if position == 1 and len(trade_records) > 0 and trade_records[-1]["卖出日期"] is None:
        last_price = df.iloc[-1]["close"]
        profit_pct = (last_price - buy_price) / buy_price
        trade_records[-1]["卖出日期"] = df.iloc[-1]["trade_date"]
        trade_records[-1]["卖出价"] = last_price
        trade_records[-1]["卖出原因"] = "回测期末强制平仓"
        trade_records[-1]["单笔收益"] = f"{profit_pct:.2%}"

    df["return"] = df["close"].pct_change()
    df = df.dropna(subset=["return"])
    if len(df) < 2:
        print("⚠️ calc_stoploss_strategy：收益率计算后数据不足！")
        return None, None, None, None
    df["strategy_return"] = df["return"] * df["signal"].shift(1)
    strategy_cum = (1 + df["strategy_return"]).cumprod()

    sell_df = pd.DataFrame(trade_records)
    stop_loss_count = (sell_df["卖出原因"] == "止损").sum() if len(sell_df) > 0 else 0
    take_profit_count = (sell_df["卖出原因"] == "止盈").sum() if len(sell_df) > 0 else 0

    metrics_dict = calc_metrics(df, df["strategy_return"])
    metrics = {
        "策略类型": f"均线+止盈止损（止损{stop_loss_pct:.0%}/止盈{take_profit_pct:.0%}）",
        "最终净值": metrics_dict["最终净值"],
        "年化收益率": metrics_dict["年化收益率"],
        "最大回撤": metrics_dict["最大回撤"],
        "夏普比率": metrics_dict["夏普比率"],
        "交易次数": len(trade_records),
        "止损次数": int(stop_loss_count),
        "止盈次数": int(take_profit_count)
    }
    return df, strategy_cum, sell_df, metrics


def main():
    parser = argparse.ArgumentParser(description="均线+止盈止损对比回测")
    parser.add_argument("--code", type=str, help="股票代码，例如600519.SH")
    parser.add_argument("--start", type=str, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, help="结束日期 YYYYMMDD")
    args = parser.parse_args()

    # 没有命令行参数，回退交互式输入
    ts_code = args.code if args.code else input("请输入股票代码（例：600519.SH）：").strip()
    start_date = args.start if args.start else input("回测开始日期(YYYYMMDD)，回车默认20230101：").strip() or "20230101"
    end_date = args.end if args.end else input("回测结束日期(YYYYMMDD)，回车默认20241231：").strip() or "20241231"

    stop_loss_pct = 0.08
    take_profit_pct = 0.15
    short_ma = 5
    long_ma = 20

    if not (start_date.isdigit() and len(start_date)==8 and end_date.isdigit() and len(end_date)==8):
        print("❌ 日期格式错误！")
        return

    df_raw = load_stock_data(ts_code, start_date, end_date)
    if df_raw is None or df_raw.empty:
        print("❌ 行情数据获取失败，程序退出")
        return

    print(f"\n回测参数：MA{short_ma}/MA{long_ma}，止损{stop_loss_pct:.0%}，止盈{take_profit_pct:.0%}")
    print("-" * 60)
    df_base, cum_base, trades_base, metrics_base = calc_base_strategy(df_raw, short_ma, long_ma)
    if df_base is None:
        print("❌ 纯均线策略数据不足，程序退出！")
        return
    print("✅ 纯均线策略回测完成")

    df_sl, cum_sl, trades_sl, metrics_sl = calc_stoploss_strategy(
        df_raw, short_ma, long_ma, stop_loss_pct, take_profit_pct
    )
    if df_sl is None:
        print("❌ 止盈止损策略数据不足，程序退出！")
        return
    print("✅ 止盈止损策略回测完成")

    print("\n" + "=" * 60)
    print(f"{ts_code} 策略对比结果")
    print("=" * 60)
    compare_df = pd.DataFrame([metrics_base, metrics_sl])
    print(compare_df.to_string(index=False))

    compare_df.to_csv("stoploss_compare_result.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 对比结果已保存：stoploss_compare_result.csv")

    if len(trades_sl) > 0:
        trades_sl.to_csv("stoploss_trade_detail.csv", index=False, encoding="utf-8-sig")
        print("✅ 止盈止损逐笔交易明细已保存：stoploss_trade_detail.csv")
        print("\n--- 止盈止损策略逐笔交易明细 ---")
        print(trades_sl.to_string(index=False))

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_base["trade_date"], cum_base.values, label="纯均线策略（无风控）", linewidth=2)
    ax.plot(df_sl["trade_date"], cum_sl.values, label="均线+止盈止损风控", linewidth=2, linestyle="--")
    benchmark_return = df_base["close"].pct_change().fillna(0)
    benchmark = (1 + benchmark_return).cumprod()
    ax.plot(df_base["trade_date"], benchmark.values, label="买入持有基准", alpha=0.5)
    ax.set_title(f"{ts_code} 纯均线 vs 止盈止损风控 净值对比\n"
                 f"(止损{stop_loss_pct:.0%} / 止盈{take_profit_pct:.0%} / MA{short_ma}-MA{long_ma})")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(alpha=0.3)
    img_name = f"{ts_code.replace('.','_')}_stoploss_compare.png"
    plt.tight_layout()
    plt.savefig(img_name, dpi=300)
    plt.show()
    print(f"\n✅ 对比图已保存：{img_name}")

    print("\n" + "=" * 60)
    print("📊 结果解读提示")
    print("=" * 60)
    print("1. 对比两条净值曲线：风控是否降低了回撤？")
    print("2. 观察止损/止盈次数：是止损触发多还是止盈触发多？")
    print("3. 思考：风控在震荡市中保护本金，但在单边上涨行情中可能减少收益")
    print("⚠️ 风险提示：回测结果仅作学习演示，不代表未来实盘表现！")


if __name__ == "__main__":
    main()