# -*- coding: utf-8 -*-
"""
上证综指日收益率波动率建模 — Part 2: 滚动预测与结论
====================================================
本模块从 renewrenew/output/ 读取 Part 1 的中间数据，完成：
  7. 滚动窗口样本外预测 (完全滚动: ARIMA + 四波动率模型)
  8. Diebold-Mariano 检验
  9. 预测对比图 + 评估指标柱状图
  10. 最终结果汇总

图表保存至 当前目录(figures/results/)，结论保存至 figures/results/。

制图规范参考 plot_figures.py：黑白学术风，A4兼容尺寸，SimSun+Times New Roman字体。
"""
import os, sys, warnings, logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sc_stats
from scipy.stats import t as t_dist, skew, kurtosis, norm as norm_dist
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

warnings.filterwarnings('ignore')

# ===== 字体与数学公式 (参考 plot_figures.py) =====
plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12

# ===== 黑白学术风配色 =====
COLOR_BLACK = '#000000'
COLOR_WHITE = '#FFFFFF'
COLOR_DARK  = '#000000'
COLOR_GRID  = '#666666'
COLOR_REF   = '#222222'

# 四模型线型区分
MODEL_STYLES = {
    'GARCH(1,1)-N':   {'ls': ':',  'lw': 1.2, 'label': 'GARCH(1,1)-N'},
    'GARCH(1,1)-t':   {'ls': '-',  'lw': 1.2, 'label': 'GARCH(1,1)-t'},
    'EGARCH(1,1)-t':  {'ls': '--', 'lw': 1.2, 'label': 'EGARCH(1,1)-t'},
    'APARCH(1,1)-t':  {'ls': '-.', 'lw': 1.2, 'label': 'APARCH(1,1)-t'},
}

MODEL_COLORS = {
    'GARCH(1,1)-N':   '#000000',
    'GARCH(1,1)-t':   '#E69F00',
    'EGARCH(1,1)-t':  '#0072B2',
    'APARCH(1,1)-t':  '#009E73',
}

MODEL_HATCHES = {
    'GARCH(1,1)-N':  '///',
    'GARCH(1,1)-t':  '\\\\\\',
    'EGARCH(1,1)-t': 'xxx',
    'APARCH(1,1)-t': '...',
}

# 指标配色
METRIC_COLORS = {
    'RMSE': '#000000',
    'MAE': '#E69F00',
    'SMAPE(%)': '#0072B2',
    'QLIKE': '#009E73',
}

# ===== A4 兼容图形尺寸 =====
SINGLE_FIG = (6.3, 6.3 * 0.618)
SINGLE_SQ  = (6.3, 6.0)
TWO_BY_TWO = (8.0, 7.5)
FOUR_ROW   = (6.3, 8.0)

# ===== 目录路径 =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # renewrenew/
RENEW_DIR = os.path.dirname(SCRIPT_DIR)                          # renew/ (当前目录)
DATA_DIR = os.path.join(SCRIPT_DIR, 'output')                    # renewrenew/output/  (读取)
FIG_DIR  = os.path.join(RENEW_DIR, 'figures', 'results')         # figures/results/    (输出)
os.makedirs(FIG_DIR, exist_ok=True)
os.chdir(SCRIPT_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(os.path.join(FIG_DIR, 'analysis_log_part2.txt'), mode='w', encoding='utf-8'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('part2_forecast')

MODEL_NAMES = ['GARCH(1,1)-N', 'GARCH(1,1)-t', 'EGARCH(1,1)-t', 'APARCH(1,1)-t']


def _savefig(fig, name):
    """统一保存 PDF 至 figures/results/"""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLOR_WHITE, edgecolor='none')
    plt.close(fig)
    logger.info(f"  图表: {name} 已保存")


# ====================== 数据加载 ======================
def load_data_from_part1():
    """从 renewrenew/output/ 加载 Part 1 保存的中间数据"""
    logger.info("=" * 60)
    logger.info("加载 Part 1 中间结果...")
    logger.info("=" * 60)

    # 处理后的数据 (含 log_return)
    proc_df = pd.read_csv(os.path.join(DATA_DIR, 'processed_data.csv'), encoding='utf-8-sig')
    r = proc_df['log_return'].values
    logger.info(f"  收益率序列: {len(r)} 个观测")

    # ARIMA 阶数
    order_df = pd.read_csv(os.path.join(DATA_DIR, 'arima_order.csv'), encoding='utf-8-sig')
    arima_order = (int(order_df['p'].iloc[0]), int(order_df['d'].iloc[0]), int(order_df['q'].iloc[0]))
    logger.info(f"  ARIMA 阶数: {arima_order}")

    # ARIMA 残差
    arima_df = pd.read_csv(os.path.join(DATA_DIR, 'arima_residuals.csv'), encoding='utf-8-sig')
    arima_resid = arima_df['arima_residual'].values
    arima_resid = arima_resid[~np.isnan(arima_resid)]
    logger.info(f"  ARIMA 残差: {len(arima_resid)} 个")

    # 标准化残差
    sr_df = pd.read_csv(os.path.join(DATA_DIR, 'standardized_residuals.csv'), encoding='utf-8-sig')
    std_resid = {}
    for name in MODEL_NAMES:
        col = name + '_std_resid'
        if col in sr_df.columns:
            vals = sr_df[col].values
            std_resid[name] = vals[~np.isnan(vals)]
        else:
            std_resid[name] = np.array([])

    # 条件波动率
    cv_df = pd.read_csv(os.path.join(DATA_DIR, 'conditional_volatility.csv'), encoding='utf-8-sig')
    cond_vol = {}
    for name in MODEL_NAMES:
        col = name + '_cond_vol'
        if col in cv_df.columns:
            vals = cv_df[col].values
            cond_vol[name] = vals[~np.isnan(vals)]
        else:
            cond_vol[name] = np.array([])

    # 描述性统计
    stats_df = pd.read_csv(os.path.join(DATA_DIR, 'descriptive_stats.csv'), encoding='utf-8-sig')
    stats_dict = {row['统计量']: row['值'] for _, row in stats_df.iterrows()}

    # 模型比较 (用于获取样本内最优)
    comp_df = pd.read_csv(os.path.join(DATA_DIR, 'model_comparison.csv'), encoding='utf-8-sig')
    best_name = comp_df.sort_values('AIC').iloc[0]['模型']
    logger.info(f"  样本内最优模型 (AIC): {best_name}")

    return r, arima_order, arima_resid, std_resid, cond_vol, stats_dict, best_name


# ====================== 7. 滚动窗口样本外预测 ======================
def rolling_forecast(r, arima_order, best_name, re_estimate_every=1):
    """
    滚动窗口样本外预测（完全滚动：每步重估 ARIMA + 波动率模型）

    Parameters
    ----------
    r : np.ndarray
        对数收益率序列（百分比尺度）
    arima_order : tuple
        ARIMA 模型的阶数 (p, d, q)
    best_name : str
        样本内最优模型名称（仅用于绘图标注）
    re_estimate_every : int
        重估频率（步数），默认值为1（每步重估）
    """
    logger.info("\n" + "=" * 60)
    logger.info("7. 滚动窗口样本外预测 (完全滚动: ARIMA + 波动率模型)")
    logger.info("=" * 60)

    T = len(r)
    n_train_init = int(T * 0.90)
    n_test = T - n_train_init
    logger.info(f"  总观测: {T}, 初始训练集: {n_train_init}, 测试集: {n_test}")
    logger.info(f"  重估频率: 每 {re_estimate_every} 步 (重新估计 ARIMA 和波动率模型)")

    model_names = list(MODEL_NAMES)
    predictions = {name: np.full(n_test, np.nan) for name in model_names}
    actual_vol = np.full(n_test, np.nan)

    last_arima_fit = None
    last_vol_fits = None

    for step in range(n_test):
        idx = n_train_init + step
        train_r = r[:idx]

        if step == 0 or step % re_estimate_every == 0:
            try:
                arima_fit = ARIMA(train_r, order=arima_order).fit()
                resid = arima_fit.resid
                if hasattr(resid, 'dropna'):
                    resid_train = resid.dropna().values
                else:
                    resid_train = resid[~np.isnan(resid)]

                if len(resid_train) < 10:
                    logger.warning(f"Step {step}: 残差长度过短 ({len(resid_train)})，跳过重估")
                    continue

                vol_fits = {}
                vol_fits['GARCH(1,1)-N'] = arch_model(resid_train, mean='Zero', vol='GARCH', p=1, q=1,
                                                       dist='normal').fit(disp='off', options={'maxiter': 1000})
                vol_fits['GARCH(1,1)-t'] = arch_model(resid_train, mean='Zero', vol='GARCH', p=1, q=1,
                                                       dist='t').fit(disp='off', options={'maxiter': 1000})
                vol_fits['EGARCH(1,1)-t'] = arch_model(resid_train, mean='Zero', vol='EGARCH', p=1, q=1,
                                                        dist='t').fit(disp='off', options={'maxiter': 1000})
                vol_fits['APARCH(1,1)-t'] = arch_model(resid_train, mean='Zero', vol='APARCH', p=1, q=1,
                                                        dist='t').fit(disp='off', options={'maxiter': 1000})

                last_arima_fit = arima_fit
                last_vol_fits = vol_fits
            except Exception as e:
                logger.warning(f"Step {step}: 模型重估失败 ({e})，跳过")
                continue

        if last_arima_fit is None or last_vol_fits is None:
            continue

        try:
            arima_forecast = last_arima_fit.forecast(steps=1).iloc[0]
        except Exception:
            arima_forecast = 0.0

        if idx < T:
            true_residual = r[idx] - arima_forecast
            actual_vol[step] = np.abs(true_residual)

        for name in model_names:
            try:
                fcast = last_vol_fits[name].forecast(horizon=1)
                var_pred = fcast.variance.values[-1, 0]
                predictions[name][step] = np.sqrt(max(var_pred, 0))
            except Exception:
                predictions[name][step] = np.nan

        if step % 10 == 0 or step == n_test - 1:
            pct = (step + 1) / n_test * 100
            n_bars = int(pct / 2)
            bar_str = '#' * n_bars + '-' * (50 - n_bars)
            sys.stdout.write(f'\r  Progress: |{bar_str}| {step + 1}/{n_test} ({pct:.1f}%)')
            sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()
    logger.info(f"  滚动预测完成 ({n_test} 步)")

    # ---------- 评估指标 ----------
    eval_all = {}
    for name in model_names:
        pred = predictions[name]
        mask = ~np.isnan(pred) & ~np.isnan(actual_vol)
        a = actual_vol[mask]; p = pred[mask]
        n_val = len(a)
        if n_val < 10:
            logger.warning(f"  {name}: 有效预测不足 ({n_val})")
            continue

        rmse = np.sqrt(np.mean((a - p)**2))
        mae = np.mean(np.abs(a - p))
        d = (np.abs(a) + np.abs(p)) / 2
        d = np.where(d == 0, 1e-10, d)
        smape = np.mean(np.abs(a - p) / d) * 100
        qlike = np.mean(np.log(p**2) + (a**2) / (p**2))

        eval_all[name] = {'RMSE': rmse, 'MAE': mae, 'SMAPE(%)': smape, 'QLIKE': qlike, 'n': n_val}
        logger.info(f"  {name} (n={n_val}): RMSE={rmse:.6f}, MAE={mae:.6f}, SMAPE={smape:.4f}%, QLIKE={qlike:.4f}")

    eval_df = pd.DataFrame([{'模型': n, **v} for n, v in eval_all.items()])
    eval_df.to_csv(os.path.join(FIG_DIR, 'forecast_evaluation_rolling.csv'), index=False, encoding='utf-8-sig')

    # ---------- Diebold-Mariano 检验 ----------
    logger.info("\n  Diebold-Mariano 检验 (平方误差损失):")
    dm_results = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            ni, nj = model_names[i], model_names[j]
            mask = ~np.isnan(predictions[ni]) & ~np.isnan(predictions[nj]) & ~np.isnan(actual_vol)
            a = actual_vol[mask]; pi = predictions[ni][mask]; pj = predictions[nj][mask]
            if len(a) < 20:
                dm_results.append({'模型1': ni, '模型2': nj, 'DM统计量': np.nan, 'DM_p值': np.nan, '结论': '样本不足'})
                continue
            ei = (a - pi)**2; ej = (a - pj)**2
            d = ei - ej
            dm_m = np.mean(d); dm_s = np.std(d, ddof=1)
            nd = len(d)
            if dm_s > 0:
                dm_stat = np.sqrt(nd) * dm_m / dm_s
                dm_p = 2 * (1 - t_dist.cdf(abs(dm_stat), df=nd - 1))
            else:
                dm_stat, dm_p = np.nan, np.nan
            sig = '***' if (not np.isnan(dm_p) and dm_p < 0.01) else ('**' if dm_p < 0.05 else ('*' if dm_p < 0.10 else ''))
            if dm_stat < 0:
                conclusion = f'{ni} 更优{sig}' if sig else f'{ni} 略优(不显著)'
            else:
                conclusion = f'{nj} 更优{sig}' if sig else f'{nj} 略优(不显著)'
            dm_results.append({'模型1': ni, '模型2': nj, 'DM统计量': dm_stat, 'DM_p值': dm_p, '结论': conclusion})
            logger.info(f"    {ni} vs {nj}: DM={dm_stat:.4f}, p={dm_p:.4f} >> {conclusion}")
    pd.DataFrame(dm_results).to_csv(os.path.join(FIG_DIR, 'dm_test_rolling.csv'), index=False, encoding='utf-8-sig')

    # ---------- 保存滚动预测数据供后续绘图复盘 ----------
    roll_data = {'actual_proxy_vol': actual_vol}
    for name in model_names:
        roll_data[name + '_pred_vol'] = predictions[name]
    pd.DataFrame(roll_data).to_csv(os.path.join(FIG_DIR, 'rolling_forecast_predictions.csv'), index=False, encoding='utf-8-sig')

    return predictions, actual_vol, eval_all, dm_results


# ====================== 图表: 滚动预测对比 ======================
def plot_rolling_forecast(predictions, actual_vol, eval_all, best_name):
    logger.info("\n[图表] 滚动预测对比图")
    fig, ax = plt.subplots(figsize=(8, 5))

    show_n = min(len(actual_vol), 250)
    if len(actual_vol) > 0 and show_n > 0:
        ax.plot(range(show_n), actual_vol[:show_n], color=COLOR_REF,
                linewidth=1.0, linestyle='-', label='|a_t| (代理)', alpha=0.7)

    for name in MODEL_NAMES:
        pred = predictions.get(name, np.array([]))
        if len(pred) == 0:
            continue
        sty = MODEL_STYLES.get(name, {'ls': '-', 'lw': 1.0})
        label = sty['label'] + (' [最优拟合]' if name == best_name else '')
        lw_use = 1.8 if name == best_name else sty['lw']
        ax.plot(range(min(show_n, len(pred))), pred[:show_n],
                color=MODEL_COLORS.get(name, COLOR_BLACK),
                linestyle=sty['ls'], linewidth=lw_use,
                alpha=0.85, label=label)

    ax.set_xlabel('预测步数 (前250步)')
    ax.set_ylabel('波动率')
    ax.set_title('滚动窗口样本外波动率预测对比 (完全滚动: ARIMA + 波动率模型)')
    ax.legend(fontsize=8, framealpha=0.9, edgecolor=COLOR_BLACK)
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
    fig.tight_layout()
    _savefig(fig, 'fig_forecast_rolling.pdf')


# ====================== 图表: 评估指标柱状图 ======================
def plot_evaluation_bar(eval_all):
    logger.info("[图表] 评估指标柱状图 (簇状分组)")

    metrics = ['RMSE', 'MAE', 'SMAPE(%)', 'QLIKE']
    metric_labels = {
        'RMSE': 'RMSE',
        'MAE': 'MAE',
        'SMAPE(%)': 'SMAPE (%)',
        'QLIKE': 'QLIKE',
    }

    data = {}
    for metric in metrics:
        data[metric] = [eval_all.get(n, {}).get(metric, np.nan) for n in MODEL_NAMES]

    n_models = len(MODEL_NAMES)
    n_metrics = len(metrics)
    bar_width = 0.18
    x = np.arange(n_models)

    all_vals = [v for metric in metrics for v in data[metric] if not np.isnan(v)]
    y_min = min(all_vals) * 0.98 if all_vals else 0
    y_max = max(all_vals) * 1.02 if all_vals else 1

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, metric in enumerate(metrics):
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        vals = data[metric]
        bars = ax.bar(x + offset, vals, bar_width,
                      color=METRIC_COLORS[metric], edgecolor=COLOR_BLACK,
                      linewidth=0.8, label=metric_labels.get(metric, metric))

        if not all(np.isnan(v) for v in vals):
            best_idx = np.nanargmin(vals)
            bars[best_idx].set_edgecolor(COLOR_BLACK)
            bars[best_idx].set_linewidth(2.5)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() - (y_max - y_min) * 0.03,
                        f'{val:.4f}', ha='center', va='top', fontsize=7.5,
                        color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('(1,1)', '') for n in MODEL_NAMES], fontsize=12)
    ax.set_ylabel('值')
    ax.set_title('滚动窗口样本外预测评估指标对比')
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor=COLOR_BLACK, ncol=4)
    ax.set_facecolor(COLOR_WHITE)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4, axis='y')

    fig.tight_layout()
    _savefig(fig, 'fig_eval_bar.pdf')


# ====================== 图表: 条件波动率对比 (复盘) ======================
def plot_conditional_volatility(cond_vol):
    logger.info("[图表] 条件波动率对比图 (半透明叠加+差异子图)")
    fig, (ax_main, ax_diff) = plt.subplots(2, 1, figsize=(8, 8),
                                            gridspec_kw={'height_ratios': [2.5, 1]})
    ref_name = 'GARCH(1,1)-N'
    ref_cv = None

    for name in MODEL_NAMES:
        cv = cond_vol.get(name, np.array([]))
        if len(cv) == 0:
            continue
        sty = MODEL_STYLES.get(name, {'ls': '-', 'lw': 1.0})
        clr = MODEL_COLORS.get(name, COLOR_BLACK)
        ax_main.plot(np.arange(len(cv)), cv, color=clr,
                     linestyle=sty['ls'], linewidth=sty['lw'],
                     alpha=0.55, label=sty['label'])
        if name == ref_name:
            ref_cv = cv
    ax_main.set_xlabel('观测序号')
    ax_main.set_ylabel('条件波动率')
    ax_main.set_title('四模型条件波动率估计对比 (α=0.55, 重叠区混合色)')
    ax_main.legend(fontsize=8, framealpha=0.9, edgecolor=COLOR_BLACK)
    ax_main.set_facecolor(COLOR_WHITE)
    ax_main.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)

    if ref_cv is not None:
        for name in MODEL_NAMES:
            if name == ref_name:
                continue
            cv = cond_vol.get(name, np.array([]))
            if len(cv) == 0:
                continue
            sty = MODEL_STYLES.get(name, {'ls': '-', 'lw': 1.0})
            clr = MODEL_COLORS.get(name, COLOR_BLACK)
            n_common = min(len(cv), len(ref_cv))
            diff = cv[:n_common] - ref_cv[:n_common]
            ax_diff.plot(np.arange(n_common), diff, color=clr,
                         linestyle=sty['ls'], linewidth=sty['lw'],
                         alpha=0.8, label=f'{name} − {ref_name}')
        ax_diff.axhline(y=0, color=COLOR_REF, linestyle='--', linewidth=0.6)
        ax_diff.set_xlabel('观测序号')
        ax_diff.set_ylabel(f'Δ 波动率 (vs {ref_name})')
        ax_diff.set_title(f'各模型与 {ref_name} 的差异')
        ax_diff.legend(fontsize=7, framealpha=0.9, edgecolor=COLOR_BLACK, ncol=3)
        ax_diff.set_facecolor(COLOR_WHITE)
        ax_diff.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)

    fig.tight_layout()
    _savefig(fig, 'fig_cond_vol.pdf')


# ====================== 图表: 标准化残差时序 ======================
def plot_std_resid_timeseries(std_resid):
    logger.info("[图表] 标准化残差时序图")
    fig, axes = plt.subplots(len(MODEL_NAMES), 1, figsize=FOUR_ROW, sharex=True)
    for idx, name in enumerate(MODEL_NAMES):
        sr = std_resid.get(name, np.array([]))
        if len(sr) == 0:
            continue
        ax = axes[idx]
        ax.plot(np.arange(len(sr)), sr, color=COLOR_BLACK, linewidth=0.4)
        ax.axhline(y=0, color=COLOR_REF, linestyle='-', linewidth=0.6)
        ax.axhline(y=3, color=COLOR_DARK, linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(y=-3, color=COLOR_DARK, linestyle='--', linewidth=0.8, alpha=0.7)
        mu_sr = np.mean(sr); sg_sr = np.std(sr)
        ax.set_title(f'{name} 标准化残差 ($\\bar{{z}}$={mu_sr:.3f}, $\\hat{{\\sigma}}$={sg_sr:.3f})', fontsize=10)
        ax.set_facecolor(COLOR_WHITE)
        ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
    axes[-1].set_xlabel('观测序号')
    fig.tight_layout()
    _savefig(fig, 'fig_std_resid_ts.pdf')


# ====================== 图表: Q-Q 图 ======================
def plot_qq(std_resid):
    logger.info("[图表] Q-Q 图 (2×2)")
    fig, axes = plt.subplots(2, 2, figsize=TWO_BY_TWO)

    for idx, name in enumerate(MODEL_NAMES):
        ax = axes[idx // 2, idx % 2]
        sr = std_resid.get(name, np.array([]))
        if len(sr) == 0:
            ax.set_title(f'{name} — 数据缺失', fontsize=10)
            continue

        (osm, osr), (slope, intercept, r) = sc_stats.probplot(sr, dist="norm")
        ax.plot(osm, osr, 'o', markersize=2.5,
                markerfacecolor=COLOR_WHITE, markeredgecolor=COLOR_BLACK,
                markeredgewidth=0.5, alpha=0.7)
        ax.plot(osm, slope * osm + intercept, color=COLOR_BLACK, linewidth=1.2)

        ax.set_xlabel('理论分位数')
        ax.set_ylabel('样本分位数')
        ax.set_title(f'{name}', fontsize=10)
        ax.set_facecolor(COLOR_WHITE)
        ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)

    fig.tight_layout()
    _savefig(fig, 'fig_qq.pdf')


# ====================== 图表: 标准化残差直方图 ======================
def plot_std_resid_histogram(std_resid):
    logger.info("[图表] 标准化残差直方图 (2×2)")
    fig, axes = plt.subplots(2, 2, figsize=TWO_BY_TWO)

    for idx, name in enumerate(MODEL_NAMES):
        ax = axes[idx // 2, idx % 2]
        sr = std_resid.get(name, np.array([]))
        if len(sr) == 0:
            continue

        ax.hist(sr, bins=80, density=True,
                facecolor=COLOR_WHITE, edgecolor=COLOR_BLACK, linewidth=0.6)

        x = np.linspace(sr.min(), sr.max(), 500)
        ax.plot(x, norm_dist.pdf(x, 0, 1), color=COLOR_BLACK, linewidth=1.2,
                linestyle='--', label='N(0,1)')

        sk = sc_stats.skew(sr)
        ku = sc_stats.kurtosis(sr, fisher=True)
        info_text = f'偏度={sk:.3f}\n超额峰度={ku:.3f}'
        ax.text(0.03, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLOR_BLACK, alpha=0.85))

        ax.set_xlabel('标准化残差')
        ax.set_ylabel('密度')
        ax.set_title(name, fontsize=13)
        ax.set_facecolor(COLOR_WHITE)
        ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.4)
        ax.legend(fontsize=10, framealpha=0.9, loc='upper right')

    fig.tight_layout()
    _savefig(fig, 'fig_std_resid_hist.pdf')


# ====================== 8. 汇总 ======================
def final_summary(stats_dict, eval_all, dm_results, best_name):
    logger.info("\n" + "=" * 60)
    logger.info("8. 最终结果汇总")
    logger.info("=" * 60)

    logger.info(f"\n  [数据概况]")
    logger.info(f"    观测数: {stats_dict.get('观测数', 'N/A')}")

    logger.info(f"\n  [样本内最优模型]")
    logger.info(f"    {best_name} (AIC 最小)")

    logger.info(f"\n  [样本外滚动预测评估]")
    best_rmse = min(eval_all, key=lambda k: eval_all[k]['RMSE'])
    best_qlike = min(eval_all, key=lambda k: eval_all[k]['QLIKE'])
    for n, v in eval_all.items():
        logger.info(f"    {n}: RMSE={v['RMSE']:.6f}, MAE={v['MAE']:.6f}, SMAPE={v['SMAPE(%)']:.4f}%, QLIKE={v['QLIKE']:.4f}")
    logger.info(f"    最佳RMSE: {best_rmse} ({eval_all[best_rmse]['RMSE']:.6f})")
    logger.info(f"    最佳QLIKE: {best_qlike} ({eval_all[best_qlike]['QLIKE']:.4f})")

    logger.info(f"\n  [Diebold-Mariano 检验结论]")
    for dm in dm_results:
        logger.info(f"    {dm['模型1']} vs {dm['模型2']}: {dm['结论']}")

    logger.info(f"\n  [核心结论]")
    logger.info(f"    1. 样本内: {best_name} 拟合最优 (AIC最小)")
    logger.info(f"    2. 样本外: {best_rmse} 预测精度最高 (RMSE最小)")
    logger.info(f"    3. APARCH-t 模型同时捕捉厚尾、杠杆效应和幂变换，提供了最灵活的波动率刻画")

    # 保存汇总
    summary = [
        {'项目': '样本内最优模型', '值': best_name},
        {'项目': '样本外最优RMSE', '值': f"{best_rmse} ({eval_all[best_rmse]['RMSE']:.6f})"},
        {'项目': '样本外最优QLIKE', '值': f"{best_qlike} ({eval_all[best_qlike]['QLIKE']:.4f})"},
        {'项目': '总观测数', '值': stats_dict.get('观测数', 'N/A')},
    ]
    pd.DataFrame(summary).to_csv(os.path.join(FIG_DIR, 'final_summary.csv'), index=False, encoding='utf-8-sig')

    logger.info(f"\n  所有结果已保存至 {FIG_DIR}/ 目录")


# ====================== 主函数 ======================
def main():
    logger.info("=" * 60)
    logger.info("上证综指波动率分析 — Part 2: 滚动预测与结论")
    logger.info("(GARCH-N, GARCH-t, EGARCH-t, APARCH-t)")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"输出目录: {FIG_DIR}")
    logger.info("=" * 60)

    # 加载 Part 1 数据
    r, arima_order, arima_resid, std_resid, cond_vol, stats_dict, best_name = \
        load_data_from_part1()

    # 7: 滚动预测 + DM 检验
    predictions, actual_vol, eval_all, dm_results = rolling_forecast(
        r, arima_order, best_name, re_estimate_every=1)

    # 图表绘制 (参考 plot_figures.py 风格)
    plot_rolling_forecast(predictions, actual_vol, eval_all, best_name)
    plot_evaluation_bar(eval_all)
    plot_conditional_volatility(cond_vol)
    plot_std_resid_timeseries(std_resid)
    plot_qq(std_resid)
    plot_std_resid_histogram(std_resid)

    # 8: 汇总
    final_summary(stats_dict, eval_all, dm_results, best_name)

    logger.info("\n" + "=" * 60)
    logger.info("Part 2 全部完成!")
    logger.info(f"图表和结论保存在: {FIG_DIR}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()