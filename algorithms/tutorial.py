import vectorbt as vbt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from dateutil.parser import parse
import ipywidgets as widgets
from copy import deepcopy
from tqdm import tqdm
import imageio
from IPython import display
import plotly.graph_objects as go
import itertools
import dateparser
import gc

# Enter your parameters here
seed = 42
symbol = 'BTC-USD'
metric = 'total_return'

start_date = datetime(2018, 1, 1, tzinfo=pytz.utc)  # time period for analysis, must be timezone-aware
end_date = datetime(2020, 1, 1, tzinfo=pytz.utc)
time_buffer = timedelta(days=100)  # buffer before to pre-calculate SMA/EMA, best to set to max window
freq = '1D'

vbt.settings.portfolio['init_cash'] = 100.  # 100$
vbt.settings.portfolio['fees'] = 0.0025  # 0.25%
vbt.settings.portfolio['slippage'] = 0.0025  # 0.25%

# Download data with time buffer
cols = ['Open', 'High', 'Low', 'Close', 'Volume']
ohlcv_wbuf = vbt.YFData.download(symbol, start=start_date - time_buffer, end=end_date).get(cols)

ohlcv_wbuf = ohlcv_wbuf.astype(np.float64)
print(ohlcv_wbuf.shape)
print(ohlcv_wbuf.columns)
fig = ohlcv_wbuf.vbt.ohlcv.plot()
fig.show()


# Create a copy of data without time buffer
wobuf_mask = (ohlcv_wbuf.index >= start_date) & (ohlcv_wbuf.index <= end_date) # mask without buffer

ohlcv = ohlcv_wbuf.loc[wobuf_mask, :]

print(ohlcv.shape)

fast_window = 30
slow_window = 80

# Pre-calculate running windows on data with time buffer
fast_ma = vbt.MA.run(ohlcv_wbuf['Open'], fast_window)
slow_ma = vbt.MA.run(ohlcv_wbuf['Open'], slow_window)

print(fast_ma.ma.shape)
print(slow_ma.ma.shape)

# Remove time buffer
fast_ma = fast_ma[wobuf_mask]
slow_ma = slow_ma[wobuf_mask]

# there should be no nans after removing time buffer
assert(~fast_ma.ma.isnull().any())
assert(~slow_ma.ma.isnull().any())

print(fast_ma.ma.shape)
print(slow_ma.ma.shape)

# Generate crossover signals
dmac_entries = fast_ma.ma_crossed_above(slow_ma)
dmac_exits = fast_ma.ma_crossed_below(slow_ma)

# Signal stats
print(dmac_entries.vbt.signals.stats(settings=dict(other=dmac_exits)))

# Build partfolio, which internally calculates the equity curve

# Volume is set to np.inf by default to buy/sell everything
# You don't have to pass freq here because our data is already perfectly time-indexed
dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits)

# Print stats
print(dmac_pf.stats())

# Now build portfolio for a "Hold" strategy
# Here we buy once at the beginning and sell at the end
hold_entries = pd.Series.vbt.signals.empty_like(dmac_entries)
hold_entries.iloc[0] = True
hold_exits = pd.Series.vbt.signals.empty_like(hold_entries)
hold_exits.iloc[-1] = True
hold_pf = vbt.Portfolio.from_signals(ohlcv['Close'], hold_entries, hold_exits)

print(hold_pf.stats())

min_window = 2
max_window = 100
perf_metrics = ['total_return', 'positions.win_rate', 'positions.expectancy', 'max_drawdown']
perf_metric_names = ['Total return', 'Win rate', 'Expectancy', 'Max drawdown']


def on_value_change(value):
    global dmac_fig

    # Calculate portfolio
    fast_window, slow_window = value['new']
    fast_ma = vbt.MA.run(ohlcv_wbuf['Open'], fast_window)
    slow_ma = vbt.MA.run(ohlcv_wbuf['Open'], slow_window)
    fast_ma = fast_ma[wobuf_mask]
    slow_ma = slow_ma[wobuf_mask]
    dmac_entries = fast_ma.ma_crossed_above(slow_ma)
    dmac_exits = fast_ma.ma_crossed_below(slow_ma)
    dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits)

    # Update figure
    if dmac_fig is None:
        dmac_fig = ohlcv['Open'].vbt.plot(trace_kwargs=dict(name='Price'))
        fast_ma.ma.vbt.plot(trace_kwargs=dict(name='Fast MA'), fig=dmac_fig)
        slow_ma.ma.vbt.plot(trace_kwargs=dict(name='Slow MA'), fig=dmac_fig)
        dmac_entries.vbt.signals.plot_as_entry_markers(ohlcv['Open'], fig=dmac_fig)
        dmac_exits.vbt.signals.plot_as_exit_markers(ohlcv['Open'], fig=dmac_fig)
    else:
        with dmac_fig.batch_update():
            dmac_fig.data[1].y = fast_ma.ma
            dmac_fig.data[2].y = slow_ma.ma
            dmac_fig.data[3].x = ohlcv['Open'].index[dmac_entries]
            dmac_fig.data[3].y = ohlcv['Open'][dmac_entries]
            dmac_fig.data[4].x = ohlcv['Open'].index[dmac_exits]
            dmac_fig.data[4].y = ohlcv['Open'][dmac_exits]
    dmac_img.value = dmac_fig.to_image(format="png")

    # Update metrics table
    sr = pd.Series([dmac_pf.deep_getattr(m) for m in perf_metrics],
                   index=perf_metric_names, name='Performance')
    metrics_html.value = sr.to_frame().style.set_properties(**{'text-align': 'right'}).render()


# Pre-calculate running windows on data with time buffer
fast_ma, slow_ma = vbt.MA.run_combs(
    ohlcv_wbuf['Open'], np.arange(min_window, max_window+1),
    r=2, short_names=['fast_ma', 'slow_ma'])

print(fast_ma.ma.shape)
print(slow_ma.ma.shape)
print(fast_ma.ma.columns)
print(slow_ma.ma.columns)

# Remove time buffer
fast_ma = fast_ma[wobuf_mask]
slow_ma = slow_ma[wobuf_mask]

print(fast_ma.ma.shape)
print(slow_ma.ma.shape)

# We perform the same steps, but now we have 4851 columns instead of 1
# Each column corresponds to a pair of fast and slow windows
# Generate crossover signals
dmac_entries = fast_ma.ma_crossed_above(slow_ma)
dmac_exits = fast_ma.ma_crossed_below(slow_ma)

print(dmac_entries.columns) # the same for dmac_exits

# Build portfolio
dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits)

# Calculate performance of each window combination
dmac_perf = dmac_pf.deep_getattr(metric)

print(dmac_perf.shape)
print(dmac_perf.index)

print(dmac_perf.idxmax())

# Convert this array into a matrix of shape (99, 99): 99 fast windows x 99 slow windows
dmac_perf_matrix = dmac_perf.vbt.unstack_to_df(symmetric=True,
    index_levels='fast_ma_window', column_levels='slow_ma_window')

print(dmac_perf_matrix.shape)

fig = dmac_perf_matrix.vbt.heatmap(
    xaxis_title='Slow window',
    yaxis_title='Fast window')
fig.show()
# remove show_svg() for interactivity