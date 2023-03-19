import numpy as np
import pandas as pd
import talib
import datetime

def datestr2dtdate(datestr):
    return datetime.datetime.strptime(datestr, '%Y-%m-%d').date()


# 获取数据

df = pd.read_csv('四种指数价格历史数据.csv').set_index('datetime')
df.index = [datestr2dtdate(e) for e in df.index]

# 计算布林带
close = df['hs300'].values
upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

# 计算布林带宽度
bandwidth = (upper - lower) / middle

# 设置阈值和交易信号
threshold = 0.05  # 布林带宽度阈值
signal = np.zeros(len(df))  # 交易信号，1为买入，-1为卖出

for i in range(1, len(df)):
    if bandwidth[i] > threshold and bandwidth[i-1] <= threshold:
        signal[i] = 1
    elif bandwidth[i] < threshold and bandwidth[i-1] >= threshold:
        signal[i] = -1

# 执行交易
position = 0  # 当前持仓
for i in range(len(df)):
    if signal[i] == 1 and position == 0:
        # 买入
        position = 1
    elif signal[i] == -1 and position == 1:
        # 卖出
        position = 0

    # 记录每天的持仓情况
    df.loc[df.index[i], 'position'] = position

# 计算收益
df['pnl'] = df['hs300'].pct_change() * df['position'].shift(1)
df['cum_pnl'] = (1 + df['pnl']).cumprod()

# 绘制收益曲线
import matplotlib.pyplot as plt

plt.plot(df['cum_pnl'])
plt.show()