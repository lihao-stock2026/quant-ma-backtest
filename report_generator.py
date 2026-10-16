"""
项目名称：完整回测报告生成器
功能：一键运行多种策略回测，汇总所有指标和交易明细，导出HTML报告
     报告包含：策略说明、指标对比表、逐笔交易明细、净值曲线图
> 风险提示：历史回测结果存在过拟合风险，不代表未来实盘收益。
"""
import argparse
import pandas as pd
import numpy as np
import warnings
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from backtest_utils import load_stock_data, calc_metrics
warnings.filterwarnings("ignore")


def run_base_backtest(df_raw, short_window=5, long_window=20):
    """纯均线策略回测"""
    df = df_raw.copy()
    df["ma_s"] = df["close"].rolling(short_window).mean()
    df["ma_l"] = df["close"].rolling(long_window).mean()
    if len(df) == 0:
        print("⚠️ run_base_backtest：均线计算后数据为空，请拉长回测时间区间！")
        return None, None, pd.DataFrame(), {}
    df = df.dropna(subset=["ma_s", "ma_l"]).reset_index(drop=True)

    df["gold"] = (df["ma_s"] > df["ma_l"]) & (df["ma_s"].shift(1) <= df["ma_l"].shift(1))
    df["dead"] = (df["ma_s"] < df["ma_l"]) & (df["ma_s"].shift(1) >= df["ma_l"].shift(1))

    df["signal"] = 0
    pos = 0
    for idx, row in df.iterrows():
        if row["gold"]:
            pos = 1
        if row["dead"]:
            pos = 0
        df.at[idx, "signal"] = pos

    df["ret"] = df["close"].pct_change()
    df = df.dropna(subset=["ret"])
    if len(df) == 0:
        print("⚠️ run_base_backtest：收益率计算后数据为空！")
        return None, None, pd.DataFrame(), {}

    df["strat_ret"] = df["ret"] * df["signal"].shift(1)
    cum = (1 + df["strat_ret"]).cumprod()

    trades = []
    buy_date, buy_price = None, 0
    for idx, row in df.iterrows():
        if row["gold"]:
            buy_date = row["trade_date"]
            buy_price = row["close"]
        if row["dead"] and buy_date is not None:
            sell_date = row["trade_date"]
            sell_price = row["close"]
            trades.append({
                "买入日期": buy_date.strftime("%Y-%m-%d"),
                "买入价": round(buy_price, 2),
                "卖出日期": sell_date.strftime("%Y-%m-%d"),
                "卖出价": round(sell_price, 2),
                "持有天数": (sell_date - buy_date).days,
                "收益率": f"{(sell_price/buy_price - 1):.2%}"
            })
            buy_date = None

    met = calc_metrics(df, df["strat_ret"])
    metrics = {
        "策略": "纯均线金叉死叉",
        "最终净值": met["最终净值"],
        "年化收益率": met["年化收益率"],
        "最大回撤": met["最大回撤"],
        "夏普比率": met["夏普比率"],
        "交易次数": len(trades)
    }
    return df, cum, pd.DataFrame(trades), metrics


def run_stoploss_backtest(df_raw, short_window=5, long_window=20,
                          stop_loss=0.08, take_profit=0.15):
    """均线+止盈止损策略"""
    df = df_raw.copy()
    df["ma_s"] = df["close"].rolling(short_window).mean()
    df["ma_l"] = df["close"].rolling(long_window).mean()
    if len(df) == 0:
        print("⚠️ run_stoploss_backtest：均线计算后数据为空，请拉长回测时间区间！")
        return None, None, pd.DataFrame(), {}
    df = df.dropna(subset=["ma_s", "ma_l"]).reset_index(drop=True)

    df["gold"] = (df["ma_s"] > df["ma_l"]) & (df["ma_s"].shift(1) <= df["ma_l"].shift(1))
    df["dead"] = (df["ma_s"] < df["ma_l"]) & (df["ma_s"].shift(1) >= df["ma_l"].shift(1))

    df["signal"] = 0
    pos = 0
    buy_price = 0
    trades = []

    for idx, row in df.iterrows():
        price = row["close"]
        if pos == 0:
            if row["gold"]:
                pos = 1
                buy_price = price
                df.at[idx, "signal"] = 1
                trades.append({
                    "买入日期": row["trade_date"].strftime("%Y-%m-%d"),
                    "买入价": round(price, 2),
                    "卖出日期": None, "卖出价": None,
                    "持有天数": None, "收益率": None, "卖出原因": "金叉买入"
                })
        else:
            profit = (price - buy_price) / buy_price
            sell_reason = None
            if profit <= -stop_loss:
                sell_reason = "止损"
            elif profit >= take_profit:
                sell_reason = "止盈"
            elif row["dead"]:
                sell_reason = "死叉平仓"

            if sell_reason:
                pos = 0
                df.at[idx, "signal"] = 0
                trades[-1].update({
                    "卖出日期": row["trade_date"].strftime("%Y-%m-%d"),
                    "卖出价": round(price, 2),
                    "持有天数": (row["trade_date"] - pd.to_datetime(trades[-1]["买入日期"])).days,
                    "收益率": f"{profit:.2%}",
                    "卖出原因": sell_reason
                })
            else:
                df.at[idx, "signal"] = 1

    df["ret"] = df["close"].pct_change()
    df = df.dropna(subset=["ret"])
    if len(df) == 0:
        print("⚠️ run_stoploss_backtest：收益率计算后数据为空！")
        return None, None, pd.DataFrame(), {}

    df["strat_ret"] = df["ret"] * df["signal"].shift(1)
    cum = (1 + df["strat_ret"]).cumprod()

    met = calc_metrics(df, df["strat_ret"])
    metrics = {
        "策略": f"均线+止盈止损(止损{stop_loss:.0%}/止盈{take_profit:.0%})",
        "最终净值": met["最终净值"],
        "年化收益率": met["年化收益率"],
        "最大回撤": met["最大回撤"],
        "夏普比率": met["夏普比率"],
        "交易次数": len(trades)
    }
    return df, cum, pd.DataFrame(trades), metrics


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_html_report(ts_code, start_date, end_date, short_ma, long_ma,
                        df_base, cum_base, trades_base, metrics_base,
                        df_sl, cum_sl, trades_sl, metrics_sl):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_base["trade_date"], cum_base.values, label="纯均线策略", linewidth=2)
    ax.plot(df_sl["trade_date"], cum_sl.values, label="均线+止盈止损", linewidth=2, linestyle="--")
    bench_series = (1 + df_base["close"].pct_change().fillna(0)).cumprod()
    ax.plot(df_base["trade_date"], bench_series.values, label="买入持有基准", alpha=0.5)
    ax.set_title(f"{ts_code} 回测净值对比（{start_date} ~ {end_date}）")
    ax.legend()
    ax.grid(alpha=0.3)
    chart_b64 = fig_to_base64(fig)
    plt.close()

    compare_df = pd.DataFrame([metrics_base, metrics_sl])
    metrics_html = compare_df.to_html(index=False, classes="metrics-table", border=0)
    trades_html = trades_sl.to_html(index=False, classes="trades-table", border=0)
    trades_base_html = trades_base.to_html(index=False, classes="trades-table", border=0)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>量化回测报告 - {ts_code}</title>
<style>
  body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
  h1 {{ color: #333; border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }}
  h2 {{ color: #2c3e50; margin-top: 30px; }}
  .meta {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
  .risk {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  .metrics-table th, .metrics-table td {{ padding: 10px; text-align: center; border-bottom: 1px solid #ddd; }}
  .metrics-table th {{ background: #2c3e50; color: white; }}
  .trades-table th, .trades-table td {{ padding: 8px; text-align: center; border-bottom: 1px solid #eee; font-size: 13px; }}
  .trades-table th {{ background: #34495e; color: white; }}
  .chart {{ text-align: center; margin: 20px 0; }}
  .chart img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #999; font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>📊 量化策略回测报告</h1>
  <div class="meta">
    <strong>股票代码：</strong>{ts_code} &nbsp;&nbsp;|&nbsp;&nbsp;
    <strong>回测区间：</strong>{start_date} ~ {end_date} &nbsp;&nbsp;|&nbsp;&nbsp;
    <strong>均线参数：</strong>MA{short_ma} / MA{long_ma}
  </div>
  <div class="risk">
    ⚠️ <strong>风险提示：</strong>本报告基于历史数据回测，存在过拟合风险，回测收益不代表未来实盘表现，不构成任何投资建议。
  </div>
  <h2>一、策略指标对比</h2>
  {metrics_html}
  <h2>二、净值曲线对比</h2>
  <div class="chart">
    < img src="data:image/png;base64,{chart_b64}" alt="净值对比图">
  </div>
  <h2>三、纯均线策略逐笔交易明细</h2>
  {trades_base_html}
  <h2>四、止盈止损策略逐笔交易明细</h2>
  {trades_html}
  <h2>五、策略说明</h2>
  <ul>
    <li><strong>纯均线策略：</strong>短期均线上穿长期均线（金叉）买入，下穿（死叉）卖出</li>
    <li><strong>止盈止损策略：</strong>金叉买入后，亏损达到8%自动止损，盈利达到15%自动止盈，死叉正常平仓</li>
    <li><strong>评价指标：</strong>年化收益率（越高越好）、最大回撤（越小风险越低）、夏普比率（单位风险超额收益）</li>
  </ul>
  <div class="footer">
    本报告由 Python + Pandas + Matplotlib 自动生成 &nbsp;|&nbsp; 仅供学习研究使用
  </div>
</div>
</body>
</html>"""
    report_file = f"report_{ts_code.replace('.','_')}.html"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html)
    return report_file


def main():
    parser = argparse.ArgumentParser(description="回测HTML报告生成器")
    parser.add_argument("--code", type=str, help="股票代码，例如600519.SH")
    parser.add_argument("--start", type=str, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, help="结束日期 YYYYMMDD")
    args = parser.parse_args()

    ts_code = args.code if args.code else input("请输入股票代码（例：600519.SH）：").strip()
    start_date = args.start if args.start else input("回测开始日期(YYYYMMDD)，回车默认20230101：").strip() or "20230101"
    end_date = args.end if args.end else input("回测结束日期(YYYYMMDD)，回车默认20241231：").strip() or "20241231"
    short_ma = int(input("短均线周期，回车默认5：").strip() or 5)
    long_ma = int(input("长均线周期，回车默认20：").strip() or 20)

    df_raw = load_stock_data(ts_code, start_date, end_date)
    if df_raw is None or df_raw.empty:
        print("❌ 行情数据获取失败，程序退出")
        return

    print("\n正在运行纯均线策略...")
    df_base, cum_base, trades_base, metrics_base = run_base_backtest(df_raw, short_ma, long_ma)
    if df_base is None:
        print("❌ 纯均线策略数据不足，程序退出！")
        return

    print("正在运行止盈止损策略...")
    df_sl, cum_sl, trades_sl, metrics_sl = run_stoploss_backtest(df_raw, short_ma, long_ma)
    if df_sl is None:
        print("❌ 止盈止损策略数据不足，程序退出！")
        return

    trades_base.to_csv("trades_base.csv", index=False, encoding="utf-8-sig")
    trades_sl.to_csv("trades_stoploss.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics_base, metrics_sl]).to_csv("metrics_compare.csv", index=False, encoding="utf-8-sig")

    report_file = generate_html_report(
        ts_code, start_date, end_date, short_ma, long_ma,
        df_base, cum_base, trades_base, metrics_base,
        df_sl, cum_sl, trades_sl, metrics_sl
    )

    print("\n" + "=" * 60)
    print("✅ 报告生成完成！")
    print("=" * 60)
    print(f"📄 HTML报告：{report_file}（双击用浏览器打开）")
    print(f"📊 纯均线交易明细：trades_base.csv")
    print(f"📊 止盈止损交易明细：trades_stoploss.csv")
    print(f"📊 指标对比表：metrics_compare.csv")
    print("\n💡 命令行示例：python report_generator.py --code 600519.SH --start 20210101 --end 20241231")


if __name__ == "__main__":
    main()