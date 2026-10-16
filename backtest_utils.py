"""
回测公共工具模块
封装：行情读取缓存、回测指标计算、异常处理
"""
import tushare as ts
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


def calc_metrics(df, strat_ret_series):
    """
    统一计算回测指标：年化、最大回撤、夏普、最终净值、交易天数
    :param df: 回测DataFrame
    :param strat_ret_series: 策略日收益率 series
    :return: dict 指标字典
    """
    ret_ser = strat_ret_series.dropna()
    if len(ret_ser) < 2:
        return {
            "最终净值": np.nan,
            "年化收益率": np.nan,
            "最大回撤": np.nan,
            "夏普比率": np.nan,
            "交易日数": len(ret_ser)
        }

    cum = (1 + ret_ser).cumprod()
    total_ret = cum.iloc[-1] - 1
    trading_days = len(ret_ser)
    year_len = trading_days / 252

    annual_ret = (1 + total_ret) ** (1 / year_len) - 1 if year_len > 0 else np.nan

    peak = cum.cummax()
    drawdown = cum / peak - 1
    max_dd = drawdown.min()

    daily_std = ret_ser.std()
    sharpe = np.nan
    if daily_std > 1e-8:
        sharpe = (ret_ser.mean() - 0.02 / 252) / daily_std * np.sqrt(252)

    return {
        "最终净值": round(cum.iloc[-1], 3),
        "年化收益率": f"{annual_ret:.2%}" if not np.isnan(annual_ret) else np.nan,
        "最大回撤": f"{max_dd:.2%}" if not np.isnan(max_dd) else np.nan,
        "夏普比率": round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "交易日数": trading_days
    }


def load_stock_data(ts_code: str, start_date: str, end_date: str):
    """
    读取股票行情，优先本地缓存，没有则从tushare拉取
    增加token、网络异常捕获
    :param ts_code: 股票代码
    :param start_date: str YYYYMMDD
    :param end_date: str YYYYMMDD
    :return: df or None
    """
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    cache_file = os.path.join(data_dir, f"{ts_code.replace('.','_')}_{start_date}_{end_date}_data.csv")

    # 读取本地缓存
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, encoding="utf-8-sig")
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").reset_index(drop=True)
            print(f"✅ 使用本地缓存文件：{cache_file}")
            return df
        except Exception as e:
            print(f"⚠️ 缓存文件读取失败，重新拉取数据: {e}")

    # 没有缓存，从tushare拉取
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        token = input("请输入你的Tushare Token：").strip()
    if not token:
        print("❌ Tushare token为空，无法拉取行情！")
        return None

    try:
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception as err:
        print(f"❌ Tushare请求异常：{err}")
        print("提示：检查token是否有效、网络是否连通")
        return None

    if df is None or len(df) == 0:
        print("❌ tushare返回空行情数据，请检查股票代码、起止日期！")
        return None

    df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    print(f"✅ 已拉取并缓存行情：{len(df)} 条K线")
    return df