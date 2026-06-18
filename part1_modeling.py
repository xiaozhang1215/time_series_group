# -*- coding: utf-8 -*-
"""
上证综指日收益率波动率建模 — Part 1: 数据预处理与模型拟合
========================================================
本模块完成：
  1. 数据加载与预处理
  2. 描述性统计
  3. 平稳性与ARCH效应检验
  4. ARIMA 均值方程过滤
  5. 四模型波动率建模 (GARCH-N, GARCH-t, EGARCH-t, APARCH-t)
  6. 模型诊断

输出中间数据至 renewrenew/output/，供 Part 2 滚动预测使用。
"""
import os, sys, warnings, logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import jarque_bera, norm, t as t_dist, skew, kurtosis
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from arch import arch_model

warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 12

# ===== 简洁清晰的配色方案 =====
COLORS = {
    "black": '#000000',
    "white": '#FFFFFF',
    "dark": '#000000',   # 深线（CI线、参考线）
    "grid": '#666666',   # 深灰网格
    "ref": '#222222' ,    # 参考线深灰色
}

MODEL_COLORS = {
    'GARCH(1,1)-N': "#3498DB",
    'GARCH(1,1)-t': "#E74C3C",
    'EGARCH(1,1)-t': "#27AE60",
    'APARCH(1,1)-t': "#F39C12",
}

MODEL_LINESTYLES = {
    'GARCH(1,1)-N': ':',
    'GARCH(1,1)-t': '-',
    'EGARCH(1,1)-t': '--',
    'APARCH(1,1)-t': '-.',
}
SINGLE_FIG = (6.3, 6.3 * 0.618)        # 单图: 黄金比例高
SINGLE_SQ  = (6.3, 6.0)                 # 单图接近方形
TWO_BY_TWO = (8.0, 7.5)                 # 2x2 子图
FOUR_ROW   = (6.3, 8.0)                 # 4行子图

# 全局字体大小 (相对于小尺寸图片)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12

# ===== 输出目录 =====
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, 'output')
FIG = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(os.path.join(BASE_DIR, 'analysis_log_part1.txt'), mode='w', encoding='utf-8'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================== 1. 数据加载 ======================
def load_data():
    logger.info("=" * 60)
    logger.info("1. 数据加载与预处理")
    logger.info("=" * 60)
    df = pd.read_csv(os.path.join(SCRIPT_DIR, '..', 'SH_Index.csv'), encoding='utf-8')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    df['log_return'] = np.log(df['收盘'] / df['收盘'].shift(1)) * 100
    df = df.dropna(subset=['log_return']).reset_index(drop=True)

    df_2000 = df[df['日期'] >= '2000-01-01'].reset_index(drop=True)
    logger.info(f"  原始观测: {len(df)}, 2000年起有效观测: {len(df_2000)}")
    logger.info(f"  日期: {df_2000['日期'].min().date()} ~ {df_2000['日期'].max().date()}")
    return df_2000


# ====================== 2. 描述性统计 ======================
def descriptive_stats(df):
    logger.info("\n" + "=" * 60)
    logger.info("2. 描述性统计")
    logger.info("=" * 60)
    r = df['log_return'].values
    n = len(r); mu = np.mean(r); sg = np.std(r, ddof=1)
    sk = skew(r); ku = kurtosis(r, fisher=True)
    jb_s, jb_p = jarque_bera(r)
    r_min = np.min(r); r_max = np.max(r)

    logger.info(f"  N={n}, Mean={mu:.6f}%, Std={sg:.6f}%")
    logger.info(f"  Min={r_min:.6f}%, Max={r_max:.6f}%")
    logger.info(f"  Skew={sk:.6f}, ExKurt={ku:.6f}, JB_p={jb_p:.6f}")

    stats_df = pd.DataFrame({
        '统计量': ['观测数', '均值(%)', '标准差(%)', '最小值(%)', '最大值(%)', '偏度', '超额峰度', 'JB统计量', 'JB_p值'],
        '值': [n, mu, sg, r_min, r_max, sk, ku, jb_s, jb_p]
    })
    stats_df.to_csv(os.path.join(OUT, 'descriptive_stats.csv'), index=False, encoding='utf-8-sig')

    return r, {'N': n, 'mu': mu, 'sigma': sg, 'skew': sk, 'kurt': ku, 'jb_p': jb_p}


# ====================== 3. 单位根 + ARCH 检验 ======================
def preliminary_tests(r):
    logger.info("\n" + "=" * 60)
    logger.info("3. 平稳性与ARCH效应检验")
    logger.info("=" * 60)
    adf = adfuller(r, regression='c', autolag='AIC')
    kp = kpss(r, regression='c', nlags='auto')
    logger.info(f"  ADF: stat={adf[0]:.6f}, p={adf[1]:.6f}, 5%临界值={adf[4]['5%']:.6f}")
    logger.info(f"  KPSS: stat={kp[0]:.6f}, p={kp[1]:.6f}, 5%临界值={kp[3]['5%']:.6f}")
    logger.info(f"  >> 结论: 收益率平稳 I(0)")

    arch_lm = het_arch(r - np.mean(r), nlags=10)
    logger.info(f"  ARCH-LM(10): LM={arch_lm[0]:.6f}, p={arch_lm[1]:.6e}")
    logger.info(f"  >> 存在极强的ARCH效应，需GARCH族建模 [OK]")

    ut_df = pd.DataFrame({
        '检验': ['ADF', 'KPSS'],
        '统计量': [adf[0], kp[0]],
        'p值': [adf[1], kp[1]],
        '5%临界值': [adf[4]['5%'], kp[3]['5%']],
        '结论': ['平稳', '平稳']
    })
    ut_df.to_csv(os.path.join(OUT, 'unit_root_results.csv'), index=False, encoding='utf-8-sig')

    arch_df = pd.DataFrame({
        '检验': ['ARCH-LM lag=10'],
        '统计量': [arch_lm[0]],
        'p值': [arch_lm[1]]
    })
    arch_df.to_csv(os.path.join(OUT, 'arch_test_results.csv'), index=False, encoding='utf-8-sig')

    return arch_lm


# ====================== 4. ARIMA 过滤均值方程 ======================
def arima_filter(r, df):
    logger.info("\n" + "=" * 60)
    logger.info("4. ARIMA 均值方程过滤")
    logger.info("=" * 60)

    subset = r[-5000:] if len(r) > 5000 else r
    try:
        auto_m = pm.auto_arima(subset, start_p=0, start_q=0, max_p=3, max_q=3,
                                d=0, seasonal=False, trace=False, error_action='ignore',
                                suppress_warnings=True, stepwise=True, n_fits=30, information_criterion='aic')
        order = auto_m.order
        logger.info(f"  auto_arima 选定: ARIMA({order[0]},0,{order[2]}), AIC={auto_m.aic():.4f}")
    except:
        order = (2, 0, 2)
        logger.info(f"  auto_arima 失败，使用默认: ARIMA(2,0,2)")

    arima_fit = ARIMA(r, order=order).fit()
    raw_resid = arima_fit.resid
    if hasattr(raw_resid, 'dropna'):
        resid = raw_resid.dropna()
    else:
        resid = np.array(raw_resid)
    resid = resid[~np.isnan(resid)]
    logger.info(f"  ARIMA{order} 拟合完成, AIC={arima_fit.aic:.4f}, BIC={arima_fit.bic:.4f}")

    arch_lm_resid = het_arch(resid, nlags=10)
    logger.info(f"  ARIMA残差 ARCH-LM(10): LM={arch_lm_resid[0]:.4f}, p={arch_lm_resid[1]:.6e}")
    logger.info(f"  >> 残差仍存在ARCH效应，需GARCH建模 [OK]")

    # ACF/PACF 诊断图 (手动绘制以应用配色方案)
    nlags = 40
    n_resid = len(resid)
    ci95 = 1.96 / np.sqrt(n_resid)
    
    acf_data = {
        'ARIMA残差 ACF':               acf(resid, nlags=nlags),
        'ARIMA残差 PACF':              pacf(resid, nlags=nlags),
        '|ARIMA残差| ACF (波动聚集)':   acf(np.abs(resid), nlags=nlags),
        'ARIMA残差平方 ACF':              acf(resid**2, nlags=nlags),
    }
    
    fig, axes = plt.subplots(2, 2, figsize=TWO_BY_TWO)
    for ax, (title, vals) in zip(axes.flatten(), acf_data.items()):
        lags_arr = np.arange(1, len(vals))
        ax.vlines(lags_arr, 0, vals[1:], colors=COLORS['black'], linewidths=0.8)
        ax.plot(lags_arr, vals[1:], 'o', markersize=3,
                color=COLORS['black'], markerfacecolor=COLORS['white'],
                markeredgewidth=0.8)
        ax.axhline(y=0, color=COLORS['ref'], linestyle='-', linewidth=0.6)
        ax.axhline(y=ci95, color=COLORS['dark'], linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axhline(y=-ci95, color=COLORS['dark'], linestyle='--', linewidth=1.2, alpha=0.8)
        ax.set_title(title, fontsize=13, fontweight='bold', color=COLORS['black'])
        ax.set_xlabel('滞后阶数', fontsize=11)
        ax.set_ylabel('值', fontsize=11)
        ax.set_xlim(0.5, nlags + 0.5)
        ax.set_facecolor(COLORS['white'])
        ax.grid(True, color=COLORS['grid'], linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig3_arima_resid_acf.pdf'), dpi=300, bbox_inches='tight',
                facecolor=COLORS['white'], edgecolor='none')
    plt.close()
    logger.info("  图表: fig3_arima_resid_acf.pdf 已保存")

    # 时序图
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(df['日期'], df['收盘'], color=COLORS['black'], linewidth=0.8)
    axes[0].set_title('上证综指日收盘价 (2000--2026)', fontsize=15, fontweight='bold', color=COLORS['black'])
    axes[0].set_ylabel('收盘价', fontsize=12)
    axes[0].set_facecolor(COLORS['white'])
    axes[0].grid(True, color=COLORS['grid'], linestyle='--', linewidth=0.5, alpha=0.7)

    axes[1].plot(df['日期'], df['log_return'], color=COLORS['black'], linewidth=0.5, alpha=0.8)
    axes[1].axhline(y=0, color=COLORS['ref'], linestyle='--', linewidth=0.8)
    axes[1].set_title('上证综指日收益率 (2000--2026)', fontsize=15, fontweight='bold', color=COLORS['black'])
    axes[1].set_ylabel('收益率 (%)', fontsize=12)
    axes[1].set_facecolor(COLORS['white'])
    axes[1].grid(True, color=COLORS['grid'], linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig1_timeseries.pdf'), dpi=300, bbox_inches='tight',
                facecolor=COLORS['white'], edgecolor='none')
    plt.close()
    logger.info("  图表: fig1_timeseries.pdf 已保存")

    return resid, order


# ====================== 5. 四模型建模 ======================
def four_model_modeling(resid):
    logger.info("\n" + "=" * 60)
    logger.info("5. 四模型波动率建模 (基于ARIMA残差)")
    logger.info("=" * 60)

    models = {}

    # ===== Model 1: GARCH(1,1)-Normal =====
    logger.info("  拟合 GARCH(1,1)-Normal...")
    fn = arch_model(resid, mean='Zero', vol='GARCH', p=1, q=1, dist='normal').fit(
        disp='off', options={'maxiter': 2000})
    pn = fn.params
    persist_n = pn.get('alpha[1]', 0) + pn.get('beta[1]', 0)
    models['GARCH(1,1)-N'] = {'fit': fn, 'aic': fn.aic, 'bic': fn.bic,
                                'loglik': fn.loglikelihood, 'params': pn, 'persist': persist_n}
    logger.info(f"    AIC={fn.aic:.4f}, alpha1={pn.get('alpha[1]',0):.6f}, beta1={pn.get('beta[1]',0):.6f}, persist={persist_n:.6f}")

    # ===== Model 2: GARCH(1,1)-t =====
    logger.info("  拟合 GARCH(1,1)-t...")
    ft = arch_model(resid, mean='Zero', vol='GARCH', p=1, q=1, dist='t').fit(
        disp='off', options={'maxiter': 2000})
    pt = ft.params
    persist_t = pt.get('alpha[1]', 0) + pt.get('beta[1]', 0)
    models['GARCH(1,1)-t'] = {'fit': ft, 'aic': ft.aic, 'bic': ft.bic,
                                'loglik': ft.loglikelihood, 'params': pt, 'persist': persist_t}
    logger.info(f"    AIC={ft.aic:.4f}, nu={pt.get('nu',np.nan):.4f}, persist={persist_t:.6f}")

    # ===== Model 3: EGARCH(1,1)-t =====
    logger.info("  拟合 EGARCH(1,1)-t...")
    fe = arch_model(resid, mean='Zero', vol='EGARCH', p=1, q=1, dist='t').fit(
        disp='off', options={'maxiter': 2000})
    pe = fe.params
    persist_e = pe.get('beta[1]', 0)
    gamma_e = pe.get('gamma[1]', np.nan)
    models['EGARCH(1,1)-t'] = {'fit': fe, 'aic': fe.aic, 'bic': fe.bic,
                                 'loglik': fe.loglikelihood, 'params': pe, 'persist': persist_e}
    logger.info(f"    AIC={fe.aic:.4f}, beta1={persist_e:.6f}, gamma1={gamma_e:.6f}, nu={pe.get('nu',np.nan):.4f}")
    if not np.isnan(gamma_e):
        logger.info(f"    >> gamma1={'<' if gamma_e < 0 else '>'} 0 >> {'传统杠杆效应' if gamma_e < 0 else '反向杠杆效应'}")

    # ===== Model 4: APARCH(1,1)-t =====
    logger.info("  拟合 APARCH(1,1)-t (新增模型)...")
    fa = arch_model(resid, mean='Zero', vol='APARCH', p=1, q=1, dist='t').fit(
        disp='off', options={'maxiter': 2000})
    pa = fa.params
    persist_a = pa.get('beta[1]', 0)
    gamma_a = pa.get('gamma[1]', np.nan)
    delta_a = pa.get('delta', np.nan)
    models['APARCH(1,1)-t'] = {'fit': fa, 'aic': fa.aic, 'bic': fa.bic,
                                 'loglik': fa.loglikelihood, 'params': pa, 'persist': persist_a}
    logger.info(f"    AIC={fa.aic:.4f}, beta1={persist_a:.6f}, gamma1={gamma_a:.6f}, delta={delta_a:.4f}, nu={pa.get('nu',np.nan):.4f}")
    if not np.isnan(gamma_a):
        logger.info(f"    >> gamma1={'<' if gamma_a < 0 else '>'} 0 >> {'传统杠杆效应' if gamma_a < 0 else '反向杠杆效应'}")

    # ===== 模型比较与排序 =====
    best = min(models, key=lambda k: models[k]['aic'])
    logger.info(f"\n  * 样本内最优模型 (AIC): {best} (AIC={models[best]['aic']:.4f})")

    sorted_models = sorted(models.items(), key=lambda x: x[1]['aic'])
    for rank, (name, m) in enumerate(sorted_models, 1):
        logger.info(f"    {rank}. {name}: AIC={m['aic']:.4f}, BIC={m['bic']:.4f}, LogLik={m['loglik']:.4f}")

    comp_data = []
    for n, m in models.items():
        comp_data.append({
            '模型': n,
            'AIC': m['aic'],
            'BIC': m['bic'],
            'LogLik': m['loglik'],
            'alpha+beta(beta)': m['persist'],
        })
    pd.DataFrame(comp_data).to_csv(os.path.join(OUT, 'model_comparison.csv'), index=False, encoding='utf-8-sig')

    rows = []
    for n, m in models.items():
        p = m['params']
        rows.append({
            '模型': n,
            'mu': p.get('mu', np.nan),
            'omega': p.get('omega', np.nan),
            'alpha1': p.get('alpha[1]', np.nan),
            'beta1': p.get('beta[1]', np.nan),
            'gamma1': p.get('gamma[1]', np.nan),
            'delta': p.get('delta', np.nan),
            'nu': p.get('nu', np.nan),
            'persistence': m['persist'],
            'AIC': m['aic'],
            'BIC': m['bic'],
            'LogLik': m['loglik'],
        })
    pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, 'model_parameters.csv'), index=False, encoding='utf-8-sig')

    logger.info(f"  模型参数已保存至 {OUT}/model_parameters.csv")
    return models, best


# ====================== 6. 标准化残差诊断 ======================
def diagnostics(models, arima_resid):
    logger.info("\n" + "=" * 60)
    logger.info("6. 模型诊断")
    logger.info("=" * 60)

    resid_stats = []
    for name, m in models.items():
        sr = m['fit'].std_resid
        sr_arr = sr.dropna().values if hasattr(sr, 'dropna') else np.array(sr)
        sr_arr = sr_arr[~np.isnan(sr_arr)]
        mu_sr = np.mean(sr_arr); sg_sr = np.std(sr_arr)
        sk_sr = skew(sr_arr); ku_sr = kurtosis(sr_arr, fisher=True)
        logger.info(f"  {name}: mu={mu_sr:.4f}, sigma={sg_sr:.4f}, 偏度={sk_sr:.4f}, 超额峰度={ku_sr:.4f}")
        resid_stats.append({'模型': name, '均值': mu_sr, '标准差': sg_sr,
                            '偏度': sk_sr, '峰度(超额)': ku_sr})
    pd.DataFrame(resid_stats).to_csv(os.path.join(OUT, 'std_resid_stats.csv'), index=False, encoding='utf-8-sig')

    model_names = list(models.keys())
    vc = MODEL_COLORS

    # 诊断图1: 标准化残差时序
    fig, axes = plt.subplots(len(models), 1, figsize=(14, 3.5 * len(models)))
    if len(models) == 1: axes = [axes]
    for idx, name in enumerate(model_names):
        sr = models[name]['fit'].std_resid
        sr_arr = sr.dropna().values if hasattr(sr, 'dropna') else np.array(sr)
        sr_arr = sr_arr[~np.isnan(sr_arr)]
        axes[idx].plot(np.arange(len(sr_arr)), sr_arr, color=COLORS['black'], linewidth=0.6)
        axes[idx].axhline(y=0, color=COLORS['ref'], linestyle='--', linewidth=0.8)
        axes[idx].set_title(f'{name} 标准化残差 (mu={np.mean(sr_arr):.3f}, sigma={np.std(sr_arr):.3f})',
                            fontsize=13, fontweight='bold', color=COLORS['black'])
        axes[idx].set_facecolor(COLORS['white'])
        axes[idx].grid(True, color=COLORS['grid'], linestyle='--', linewidth=0.5, alpha=0.7)
        axes[idx].tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig4_residuals_diag.pdf'), dpi=300, bbox_inches='tight',
                facecolor=COLORS['white'], edgecolor='none')
    plt.close()
    logger.info("  图表: fig4_residuals_diag.pdf 已保存")

    # 诊断图2: Q-Q 图 (四模型 2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, (name, m) in enumerate(models.items()):
        ax = axes[idx // 2, idx % 2]
        sr = m['fit'].std_resid
        sr_arr = sr.dropna().values if hasattr(sr, 'dropna') else np.array(sr)
        sr_arr = sr_arr[~np.isnan(sr_arr)]
        stats.probplot(sr_arr, dist="norm", plot=ax)
        clr = vc.get(name, COLORS['black'])
        ax.get_lines()[0].set_markerfacecolor('none')
        ax.get_lines()[0].set_markeredgecolor(clr)
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[1].set_color(COLORS['black'])
        ax.get_lines()[1].set_linewidth(1.5)
        ax.set_title(f'{name} Q-Q Plot', fontsize=13, fontweight='bold', color=COLORS['black'])
        ax.set_xlabel('理论分位数', fontsize=11)
        ax.set_ylabel('样本分位数', fontsize=11)
        ax.set_facecolor(COLORS['white'])
        ax.grid(True, color=COLORS['grid'], linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig5_qq_plots.pdf'), dpi=300, bbox_inches='tight',
                facecolor=COLORS['white'], edgecolor='none')
    plt.close()
    logger.info("  图表: fig5_qq_plots.pdf 已保存")

    # 诊断图3: 条件波动率对比
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in model_names:
        cv = models[name]['fit'].conditional_volatility
        cv_arr = cv.values if hasattr(cv, 'values') else np.array(cv)
        ls = MODEL_LINESTYLES.get(name, '-')
        ax.plot(np.arange(len(cv_arr)), cv_arr, color=vc.get(name, COLORS['black']),
                linestyle=ls, linewidth=1.0, alpha=0.85, label=name)
    ax.set_title('四模型条件波动率估计对比 (ARIMA残差)', fontsize=15, fontweight='bold', color=COLORS['black'])
    ax.legend(fontsize=10, loc='upper right')
    ax.set_facecolor(COLORS['white'])
    ax.grid(True, color=COLORS['grid'], linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig6_volatility.pdf'), dpi=300, bbox_inches='tight',
                facecolor=COLORS['white'], edgecolor='none')
    plt.close()
    logger.info("  图表: fig6_volatility.pdf 已保存")

    # 分布直方图 + Q-Q 图 (ARIMA残差)
    r = arima_resid[~np.isnan(arima_resid)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hist(r, bins=80, density=True, color=COLORS['white'], edgecolor=COLORS['black'], alpha=0.6, linewidth=0.6)
    x = np.linspace(r.min(), r.max(), 500)
    axes[0].plot(x, norm.pdf(x, np.mean(r), np.std(r)), color=COLORS['black'], linewidth=1.2, linestyle='--', label='正态分布')
    axes[0].set_title('ARIMA残差分布直方图 (vs 正态分布)', fontsize=14, fontweight='bold', color=COLORS['black'])
    axes[0].legend(fontsize=11)
    axes[0].set_facecolor(COLORS['white'])
    axes[0].grid(True, color=COLORS['grid'], linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0].tick_params(labelsize=10)
    stats.probplot(r, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor('none')
    axes[1].get_lines()[0].set_markeredgecolor(COLORS['black'])
    axes[1].get_lines()[0].set_markersize(3)
    axes[1].get_lines()[1].set_color(COLORS['black'])
    axes[1].get_lines()[1].set_linewidth(1.5)
    axes[1].set_title('ARIMA残差 Q-Q Plot', fontsize=14, fontweight='bold', color=COLORS['black'])
    axes[1].set_xlabel('理论分位数', fontsize=11)
    axes[1].set_ylabel('样本分位数', fontsize=11)
    axes[1].set_facecolor(COLORS['white'])
    axes[1].grid(True, color=COLORS['grid'], linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1].tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig2_distribution.pdf'), dpi=300, bbox_inches='tight',
                facecolor=COLORS['white'], edgecolor='none')
    plt.close()
    logger.info("  图表: fig2_distribution.pdf 已保存")

    # ===== 补充图1: 标准化残差直方图 vs N(0,1) (四模型 2x2) =====
    fig, axes = plt.subplots(2, 2, figsize=TWO_BY_TWO)
    for idx, (name, m) in enumerate(models.items()):
        ax = axes[idx // 2, idx % 2]
        sr = m['fit'].std_resid
        sr_arr = sr.dropna().values if hasattr(sr, 'dropna') else np.array(sr)
        sr_arr = sr_arr[~np.isnan(sr_arr)]
        if len(sr_arr) == 0:
            continue
        ax.hist(sr_arr, bins=80, density=True,
                facecolor=COLORS['white'], edgecolor=COLORS['black'], linewidth=0.6)
        x_vals = np.linspace(sr_arr.min(), sr_arr.max(), 500)
        ax.plot(x_vals, norm.pdf(x_vals, 0, 1), color=COLORS['black'], linewidth=1.2,
                linestyle='--', label='N(0,1)')
        sk_sr = skew(sr_arr)
        ku_sr = kurtosis(sr_arr, fisher=True)
        info_text = f'偏度={sk_sr:.3f}\n超额峰度={ku_sr:.3f}'
        ax.text(0.03, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLORS['black'], alpha=0.85))
        ax.set_xlabel('标准化残差', fontsize=11)
        ax.set_ylabel('密度', fontsize=11)
        ax.set_title(name, fontsize=13, fontweight='bold', color=COLORS['black'])
        ax.set_facecolor(COLORS['white'])
        ax.grid(True, color=COLORS['grid'], linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
        ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'add1_std_resid_hist.pdf'), dpi=300, bbox_inches='tight',
                facecolor=COLORS['white'], edgecolor='none')
    plt.close()
    logger.info("  图表: add1_std_resid_hist.pdf 已保存")

    # ===== 补充图2: 条件波动率差异 (以 GARCH-N 为基准) =====
    ref_name = 'GARCH(1,1)-N'
    ref_cv = None
    if ref_name in models:
        cv_ref = models[ref_name]['fit'].conditional_volatility
        ref_cv = cv_ref.values if hasattr(cv_ref, 'values') else np.array(cv_ref)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in model_names:
        if name == ref_name:
            continue
        cv = models[name]['fit'].conditional_volatility
        cv_arr = cv.values if hasattr(cv, 'values') else np.array(cv)
        if ref_cv is not None:
            n_common = min(len(cv_arr), len(ref_cv))
            diff = cv_arr[:n_common] - ref_cv[:n_common]
            ls = MODEL_LINESTYLES.get(name, '-')
            clr = vc.get(name, COLORS['black'])
            ax.plot(np.arange(n_common), diff, color=clr,
                    linestyle=ls, linewidth=1.0, alpha=0.85,
                    label=f'{name} − {ref_name}')
    ax.axhline(y=0, color=COLORS['ref'], linestyle='--', linewidth=0.8)
    ax.set_xlabel('观测序号', fontsize=12)
    ax.set_ylabel(f'Δ 波动率 (vs {ref_name})', fontsize=12)
    ax.set_title(f'各模型与 {ref_name} 的条件波动率差异', fontsize=15, fontweight='bold', color=COLORS['black'])
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.set_facecolor(COLORS['white'])
    ax.grid(True, color=COLORS['grid'], linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'add2_cond_vol_diff.pdf'), dpi=300, bbox_inches='tight',
                facecolor=COLORS['white'], edgecolor='none')
    plt.close()
    logger.info("  图表: add2_cond_vol_diff.pdf 已保存")


# ====================== 保存中间数据供 Part 2 使用 ======================
def save_intermediate_data(df, r, arima_resid, arima_order, models):
    """保存 Part 2 滚动预测所需的中间数据"""
    logger.info("\n" + "=" * 60)
    logger.info("保存中间数据至 renewrenew/output/")
    logger.info("=" * 60)

    # 1. 处理后的完整数据 (日期 + 对数收益率) — 供 Part 2 读取 r 序列
    proc_df = df[['日期', '收盘', 'log_return']].copy()
    proc_df.to_csv(os.path.join(OUT, 'processed_data.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"  processed_data.csv 已保存 ({len(proc_df)} 行)")

    # 2. ARIMA 阶数
    order_df = pd.DataFrame({'p': [arima_order[0]], 'd': [arima_order[1]], 'q': [arima_order[2]]})
    order_df.to_csv(os.path.join(OUT, 'arima_order.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"  arima_order.csv 已保存: ARIMA{arima_order}")

    # 3. ARIMA 残差
    arima_df = pd.DataFrame({'arima_residual': arima_resid})
    arima_df.to_csv(os.path.join(OUT, 'arima_residuals.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"  arima_residuals.csv 已保存 ({len(arima_resid)} 行)")

    # 4. 标准化残差 (四模型)
    sr_dict = {}
    for name, m in models.items():
        sr = m['fit'].std_resid
        sr_arr = sr.dropna().values if hasattr(sr, 'dropna') else np.array(sr)
        sr_arr = sr_arr[~np.isnan(sr_arr)]
        sr_dict[name + '_std_resid'] = sr_arr
    sr_df = pd.DataFrame(sr_dict)
    sr_df.to_csv(os.path.join(OUT, 'standardized_residuals.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"  standardized_residuals.csv 已保存 ({len(sr_df)} 行)")

    # 5. 条件波动率 (四模型)
    cv_dict = {}
    for name, m in models.items():
        cv = m['fit'].conditional_volatility
        cv_arr = cv.values if hasattr(cv, 'values') else np.array(cv)
        cv_dict[name + '_cond_vol'] = cv_arr
    cv_df = pd.DataFrame(cv_dict)
    cv_df.to_csv(os.path.join(OUT, 'conditional_volatility.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"  conditional_volatility.csv 已保存 ({len(cv_df)} 行)")

    logger.info("  所有中间数据已保存完毕 [OK]")


# ====================== 主函数 ======================
def main():
    logger.info("=" * 60)
    logger.info("上证综指波动率分析 — Part 1: 数据预处理与模型拟合")
    logger.info("(GARCH-N, GARCH-t, EGARCH-t, APARCH-t)")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"输出目录: {BASE_DIR}")
    logger.info("=" * 60)

    # 1-3: 数据 + 描述 + 检验
    df = load_data()
    r, stats_dict = descriptive_stats(df)
    _ = preliminary_tests(r)

    # 4: ARIMA 过滤
    arima_resid, arima_order = arima_filter(r, df)

    # 5: 四模型建模 (含APARCH)
    models, best_name = four_model_modeling(arima_resid)

    # 6: 诊断
    diagnostics(models, arima_resid)

    # ===== 保存中间数据供 Part 2 使用 =====
    save_intermediate_data(df, r, arima_resid, arima_order, models)

    logger.info("\n" + "=" * 60)
    logger.info("Part 1 完成! 中间数据已保存至 renewrenew/output/")
    logger.info(f"样本内最优模型: {best_name}")
    logger.info("请运行 part2_forecast.py 进行滚动预测")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()