"""
MIT License

Copyright (c) 2021 RomFR57

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""

from math import fabs
import numpy as np
from numba import jit
from numba.extending import overload


def calculate_indicators(mean_, close_, open_, high_, low_, volume_):
    indicators = {}

    indicators['close'] = close_
    indicators['open'] = open_
    indicators['high'] = high_
    indicators['low'] = low_
    indicators['volume'] = volume_

    short_wma = wma(data=mean_, period=20)
    indicators['short_wma'] = short_wma
    medium_wma = wma(data=mean_, period=50)
    indicators['wma'] = medium_wma
    long_wma = wma(data=mean_, period=100)
    indicators['long_wma'] = long_wma

    sma_ = sma(mean_, period=20)
    indicators['sma'] = sma_
    ema_ = ema(mean_, period=20)
    indicators['ema'] = ema_

    ewma_ = ewma(data=mean_, period=15, alpha=0.97)
    indicators['ewma'] = ewma_
    tema_ = trix(data=mean_, period=30)
    indicators['tema'] = tema_
    macd_ = macd(data=mean_, fast=12, slow=26)
    indicators['macd'] = macd_
    stoch1, stoch2 = stoch(close_, high_, low_, period_k=5, period_d=3)
    indicators['stoch1'] = stoch1
    indicators['stoch2'] = stoch2
    wpr_ = wpr(close_, high_, low_, 14)
    indicators['wpr'] = wpr_
    rsi_experts = rsi(mean_, period=5)
    indicators['rsi_experts'] = rsi_experts
    rsi_ = rsi(mean_, period=14)
    indicators['rsi'] = rsi_
    srsi_ = srsi(mean_, period=14)
    indicators['srsi'] = srsi_
    bollinger_mid, bolinger_up, bolinger_down, _ = bollinger_bands(mean_, period=20)
    indicators['bollinger'] = bollinger_mid
    indicators['bolinger_up'] = bolinger_up
    indicators['bolinger_down'] = bolinger_down
    _, kc_up, kc_down, _ = keltner_channel(close_, open_, high_, low_, period=20)
    indicators['kc_up'] = kc_up
    indicators['kc_down'] = kc_down
    tenkansen, kinjunsen, chikou, senkou_a, senkou_b = ichimoku(mean_, tenkansen=9, kinjunsen=26, senkou_b=52, shift=26)
    indicators['tenkansen'] = tenkansen
    indicators['kinjunsen'] = kinjunsen
    indicators['chikou'] = chikou
    indicators['senkou_a'] = senkou_a
    indicators['senkou_b'] = senkou_b
    atr_ = atr(open_, high_, low_, period=14)
    indicators['atr'] = atr_
    momentum_ = momentum(mean_, period=40)
    indicators['momentum_'] = momentum_
    roc_ = roc(mean_, period=12)
    indicators['roc'] = roc_
    vix_ = vix(close_, low_, period=30)
    indicators['vix'] = vix_
    chop_ = chop(close_, open_, high_, low_)
    indicators['chop'] = chop_
    cog_ = cog(mean_)
    indicators['cog'] = cog_

    obv_ = on_balance_volume(close_, volume_)
    indicators['obv'] = obv_

    return indicators


def on_balance_volume(close_, volume_):
    obv = np.zeros(len(close_))
    if len(obv) == 0:
        return obv
    obv[0] = volume_[0]
    for i in range(1, len(close_)):
        if close_[i] > close_[i - 1]:
            obv[i] = obv[i - 1] + volume_[i]
        elif close_[i] < close_[i - 1]:
            obv[i] = obv[i - 1] - volume_[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def add_indicators_to_dataset(indicators, indicators_names, dates, mean_):
    new_data = []
    for i in range(len(indicators_names)):
        item = indicators_names[i]
        new_data.append(indicators[item])
    new_data.append(mean_)
    indicators_names.append('mean')
    new_data = np.array(new_data)
    new_data = np.swapaxes(new_data, 0, 1)
    new_data = new_data[100:, :]
    new_dates = dates[100:]
    return new_data, new_dates


# ✅ النسخة النهائية الصحيحة لـ np_clip (تصلح الخطأ AttributeError)
@overload(np.clip)
def np_clip(a, a_min, a_max, out=None):
    """
    Safe Numba Overload of np.clip
    """
    def impl(a, a_min, a_max, out=None):
        if out is None:
            out = np.empty_like(a)
        n = len(a)
        for i in range(n):
            val = a[i]
            if val < a_min:
                out[i] = a_min
            elif val > a_max:
                out[i] = a_max
            else:
                out[i] = val
        return out
    return impl


@jit(nopython=True)
def convolve(data, kernel):
    size_data = len(data)
    size_kernel = len(kernel)
    size_out = size_data - size_kernel + 1
    out = np.array([np.nan] * size_out)
    kernel = np.flip(kernel)
    for i in range(size_out):
        window = data[i:i + size_kernel]
        out[i] = sum([window[j] * kernel[j] for j in range(size_kernel)])
    return out


@jit(nopython=True)
def sma(data, period):
    size = len(data)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        window = data[i - period + 1:i + 1]
        out[i] = np.mean(window)
    return out


@jit(nopython=True)
def wma(data, period):
    weights = np.arange(period, 0, -1)
    weights = weights / weights.sum()
    out = convolve(data, weights)
    return np.concatenate((np.array([np.nan] * (len(data) - len(out))), out))


@jit(nopython=True)
def ema(data, period, smoothing=2.0):
    size = len(data)
    weight = smoothing / (period + 1)
    out = np.array([np.nan] * size)
    out[0] = data[0]
    for i in range(1, size):
        out[i] = (data[i] * weight) + (out[i - 1] * (1 - weight))
    out[:period - 1] = np.nan
    return out


@jit(nopython=True)
def ewma(data, period, alpha=1.0):
    weights = (1 - alpha) ** np.arange(period)
    weights /= np.sum(weights)
    out = convolve(data, weights)
    return np.concatenate((np.array([np.nan] * (len(data) - len(out))), out))


@jit(nopython=True)
def trix(data, period, smoothing=2.0):
    return ((3 * ema(data, period, smoothing) - (3 * ema(ema(data, period, smoothing), period, smoothing))) +
            ema(ema(ema(data, period, smoothing), period, smoothing), period, smoothing))


@jit(nopython=True)
def macd(data, fast, slow, smoothing=2.0):
    return ema(data, fast, smoothing) - ema(data, slow, smoothing)


@jit(nopython=True)
def stoch(c_close, c_high, c_low, period_k, period_d):
    size = len(c_close)
    k = np.array([np.nan] * size)
    for i in range(period_k - 1, size):
        e = i + 1
        s = e - period_k
        ml = np.min(c_low[s:e])
        if ml == np.max(c_high[s:e]):
            ml = ml - 0.1
        k[i] = ((c_close[i] - ml) / (np.max(c_high[s:e]) - ml)) * 100
    return k, sma(k, period_d)


@jit(nopython=True)
def wpr(c_close, c_high, c_low, period):
    size = len(c_close)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        e = i + 1
        s = e - period
        mh = np.max(c_high[s:e])
        out[i] = ((mh - c_close[i]) / (mh - np.min(c_low[s:e]))) * -100
    return out


@jit(nopython=True)
def rsi(data, period, smoothing=2.0, f_sma=True, f_clip=True, f_abs=True):
    size = len(data)
    delta = np.diff(data)
    if f_clip:
        up, down = np.clip(delta, a_min=0, a_max=np.max(delta)), np.clip(delta, a_min=np.min(delta), a_max=0)
    else:
        up, down = delta.copy(), delta.copy()
        up[delta < 0] = 0.0
        down[delta > 0] = 0.0
    if f_abs:
        for i, x in enumerate(down):
            down[i] = fabs(x)
    else:
        down = np.abs(down)
    rs = sma(up, period) / sma(down, period) if f_sma else ema(up, period - 1, smoothing) / ema(
        down, period - 1, smoothing)
    out = np.array([np.nan] * size)
    out[1:] = (100 - 100 / (1 + rs))
    return out


@jit(nopython=True)
def srsi(data, period):
    r = rsi(data, period)[period:]
    s = np.array([np.nan] * len(r))
    for i in range(period - 1, len(r)):
        window = r[i + 1 - period:i + 1]
        mw = np.min(window)
        s[i] = ((r[i] - mw) / (np.max(window) - mw)) * 100
    return np.concatenate((np.array([np.nan] * (len(data) - len(s))), s))


@jit(nopython=True)
def bollinger_bands(data, period, dev_up=2.0, dev_down=2.0):
    size = len(data)
    bb_up = np.array([np.nan] * size)
    bb_down = np.array([np.nan] * size)
    bb_width = np.array([np.nan] * size)
    bb_mid = sma(data, period)
    for i in range(period - 1, size):
        std_dev = np.std(data[i - period + 1:i + 1])
        mid = bb_mid[i]
        bb_up[i] = mid + (std_dev * dev_up)
        bb_down[i] = mid - (std_dev * dev_down)
        bb_width[i] = bb_up[i] - bb_down[i]
    return bb_mid, bb_up, bb_down, bb_width


@jit(nopython=True)
def keltner_channel(c_close, c_open, c_high, c_low, period, smoothing=2.0):
    e = ema(c_close, period, smoothing)
    aa = 2 * atr(c_open, c_high, c_low, period)
    up = e + aa
    down = e - aa
    return e, up, down, up - down


@jit(nopython=True)
def atr(c_open, c_high, c_low, period):
    return sma(np.maximum(np.maximum(c_open - c_low, np.abs(c_high - c_open)), np.abs(c_low - c_open)), period)


@jit(nopython=True)
def vix(c_close, c_low, period):
    size = len(c_close)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        hc = np.max(c_close[i + 1 - period:i + 1])
        out[i] = ((hc - c_low[i]) / hc) * 100
    return out


@jit(nopython=True)
def momentum(data, period):
    size = len(data)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        out[i] = data[i] - data[i - period + 1]
    return out


@jit(nopython=True)
def roc(data, period):
    size = len(data)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        p = data[i - period + 1]
        out[i] = ((data[i] - p) / p) * 100
    return out


@jit(nopython=True)
def chop(c_close, c_open, c_high, c_low, period=14):
    size = len(c_close)
    out = np.array([np.nan] * size)
    a_tr = atr(c_open, c_high, c_low, period)
    for i in range(period - 1, size):
        e = i + 1
        s = e - period
        out[i] = (100 * np.log10(np.sum(a_tr[s:e]) / (np.max(c_high[s:e]) - np.min(c_low[s:e])))) / np.log10(period)
    return out


@jit(nopython=True)
def cog(data, period=10):
    size = len(data)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        e = i + 1
        s = e - period
        window = data[s:e]
        den = np.sum(window)
        num = 0
        for j in range(period):
            num += window[j] * (period - j)
        out[i] = - num / den
    return out


@jit(nopython=True)
def ichimoku(data, tenkansen=9, kinjunsen=26, senkou_b=52, shift=26):
    """
    Ichimoku Cloud indicator
    :type data: np.ndarray
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    size = len(data)
    n_tenkansen = np.array([np.nan] * size)
    n_kinjunsen = np.array([np.nan] * size)
    n_senkou_b = np.array([np.nan] * (size + shift))

    for i in range(tenkansen - 1, size):
        window = data[i + 1 - tenkansen:i + 1]
        n_tenkansen[i] = (np.max(window) + np.min(window)) / 2

    for i in range(kinjunsen - 1, size):
        window = data[i + 1 - kinjunsen:i + 1]
        n_kinjunsen[i] = (np.max(window) + np.min(window)) / 2

    for i in range(senkou_b - 1, size):
        window = data[i + 1 - senkou_b:i + 1]
        n_senkou_b[i + shift] = (np.max(window) + np.min(window)) / 2

    chikou = np.concatenate((data[shift:], np.array([np.nan] * shift)))
    senkou_a = np.concatenate((np.array([np.nan] * shift), (n_tenkansen + n_kinjunsen) / 2))

    return n_tenkansen, n_kinjunsen, chikou, senkou_a, n_senkou_b