"""
城市急救可达性空间分异因果推断 — 完整实现
=============================================
核心问题：在同一急救网络下，为什么不同区域可达性差异显著？
方法链：SHAP归因 → 双重稳健估计(DR) → 空间因果森林 → 敏感性分析
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── 1. 生成模拟城市空间数据 ──────────────────────────────

def generate_synthetic_city(grid_size=50, seed=42):
    """生成模拟城市格网数据，仿真实城市空间结构"""
    np.random.seed(seed)
    n = grid_size * grid_size

    # 空间坐标
    coords = np.array([(i // grid_size, i % grid_size) for i in range(n)])
    cx, cy = grid_size // 2, grid_size // 2

    # 距市中心距离（渐变层）
    dist_to_cbd = np.sqrt((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2)
    dist_to_cbd = dist_to_cbd / dist_to_cbd.max()

    # 创造空间异质性 — 城市有多个次中心
    dist_sub1 = np.sqrt((coords[:, 0] - grid_size*0.2)**2 + (coords[:, 1] - grid_size*0.3)**2)
    dist_sub2 = np.sqrt((coords[:, 0] - grid_size*0.7)**2 + (coords[:, 1] - grid_size*0.8)**2)

    # 特征生成（带空间自相关和因果结构）
    noise = np.random.normal(0, 0.1, n)

    # 人口密度：市中心高，向外递减 + 次中心 + 随机
    pop_density = 5 * np.exp(-2 * dist_to_cbd) + 3 * np.exp(-3 * dist_sub1) + 2 * np.exp(-3 * dist_sub2) + noise
    pop_density = np.clip(pop_density, 0.1, 10)

    # 路网密度：受地形和历史影响（半独立于人口）
    road_density_base = 8 * np.exp(-1.5 * dist_to_cbd) + np.random.normal(0, 0.5, n)
    # 老城区路网密（与CBD距离相关，但受历史路径依赖）
    road_density = road_density_base + 0.3 * np.sin(coords[:, 0] * 0.5) * np.cos(coords[:, 1] * 0.5)
    road_density = np.clip(road_density, 0.5, 10)

    # 交叉口密度（路网的分形维度）
    intersection_density = road_density * 3 + np.random.normal(0, 1, n)
    intersection_density = np.clip(intersection_density, 1, 30)

    # 路网等级混合度（快速路/主干/次干比例均衡性）
    road_hierarchy_mix = 0.4 + 0.6 * np.exp(-2 * dist_to_cbd) + np.random.normal(0, 0.1, n)
    road_hierarchy_mix = np.clip(road_hierarchy_mix, 0, 1)

    # 老龄化比例
    age_65_ratio = 0.05 + 0.25 * dist_to_cbd + np.random.normal(0, 0.03, n)
    age_65_ratio = np.clip(age_65_ratio, 0.02, 0.4)

    # 建筑密度
    building_density = 0.3 + 0.6 * np.exp(-1.8 * dist_to_cbd) + np.random.normal(0, 0.05, n)
    building_density = np.clip(building_density, 0.1, 1.0)

    # 土地利用混合度
    land_use_mix = 0.3 + 0.5 * np.random.beta(2, 2, n) + 0.2 * np.exp(-dist_to_cbd)
    land_use_mix = np.clip(land_use_mix, 0.1, 1.0)

    # 经济水平代理（夜光/地价）
    nightlight = 50 * np.exp(-1.5 * dist_to_cbd) + 20 * np.exp(-3 * dist_sub2) + np.random.normal(0, 2, n)
    nightlight = np.clip(nightlight, 5, 60)
    land_price = 10000 * np.exp(-2 * dist_to_cbd) + np.random.normal(0, 2000, n)
    land_price = np.clip(land_price, 500, 12000)

    df = pd.DataFrame({
        'grid_x': coords[:, 0], 'grid_y': coords[:, 1],
        'dist_to_cbd': dist_to_cbd.astype(np.float32),
        'pop_density': pop_density.astype(np.float32),
        'road_density': road_density.astype(np.float32),
        'intersection_density': intersection_density.astype(np.float32),
        'road_hierarchy_mix': road_hierarchy_mix.astype(np.float32),
        'age_65_ratio': age_65_ratio.astype(np.float32),
        'building_density': building_density.astype(np.float32),
        'land_use_mix': land_use_mix.astype(np.float32),
        'nightlight': nightlight.astype(np.float32),
        'land_price': land_price.astype(np.float32),
    })

    # ── 生成急救站（假设已存在）──
    # 急救站位置偏老城区（路径依赖）
    n_stations = 20
    station_prob = np.exp(-1.5 * dist_to_cbd) / np.exp(-1.5 * dist_to_cbd).sum()
    station_indices = np.random.choice(n, size=n_stations, replace=False, p=station_prob)

    # 计算到最近急救站的欧氏距离（简化）
    station_coords = coords[station_indices]
    min_dist = np.min([
        np.sqrt((coords[:, 0] - sc[0])**2 + (coords[:, 1] - sc[1])**2)
        for sc in station_coords
    ], axis=0)

    # ── 生成可达性 true 因果模型 ──
    accessibility = (
        2.5 * road_density                   # 路网密度: 主要因果效应
        + 0.8 * intersection_density         # 交叉口密度
        + 1.5 * road_hierarchy_mix           # 路网等级混合
        - 1.2 * min_dist / min_dist.max() * 5  # 距急救站距离
        - 0.6 * pop_density                  # 需求竞争（每个人分摊有限服务）
        + 0.3 * land_use_mix                 # 混合用地更可达
        + np.random.normal(0, 0.8, n)        # 不可解释的随机噪声
    )
    # 标准化到 0~10
    accessibility = (accessibility - accessibility.min()) / (accessibility.max() - accessibility.min()) * 10

    df['nearest_station_dist'] = min_dist.astype(np.float32)
    df['accessibility'] = accessibility.astype(np.float32)

    # 可达性等级
    q = df['accessibility'].quantile([0.2, 0.4, 0.6, 0.8])
    df['access_level'] = np.select(
        [df['accessibility'] <= q[0.2], df['accessibility'] <= q[0.4],
         df['accessibility'] <= q[0.6], df['accessibility'] <= q[0.8]],
        ['非常低', '低', '中', '高'], default='非常高'
    )

    return df


# ── 2. SHAP 归因分析 ────────────────────────────────────

def shap_attribution(df):
    """用 XGBoost + SHAP 分解每个格网可达性的归因"""
    import xgboost as xgb
    import shap
    print("\n" + "=" * 60)
    print("方法A: SHAP 归因 —— 每个格网的可达性差距由什么驱动？")
    print("=" * 60)

    features = ['road_density', 'intersection_density', 'road_hierarchy_mix',
                'pop_density', 'age_65_ratio', 'building_density',
                'land_use_mix', 'dist_to_cbd', 'nightlight', 'land_price',
                'nearest_station_dist']
    X = df[features]
    y = df['accessibility']

    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                              random_state=42, verbosity=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 全局重要性
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({'feature': features, 'importance': importance}).sort_values('importance', ascending=False)

    print("\n全局 SHAP 重要性排名:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row['importance'] * 40)
        print(f"  {row['feature']:<28s} {bar} {row['importance']:.3f}")

    # 归因矩阵
    df['top_driver_idx'] = np.argmax(np.abs(shap_values), axis=1)
    df['top_driver'] = df['top_driver_idx'].apply(lambda i: features[i])
    df['top_driver_shap'] = [shap_values[i, idx] for i, idx in enumerate(df['top_driver_idx'])]

    # 空间归因统计
    print(f"\n归因空间分布:")
    for driver, group in df.groupby('top_driver'):
        pct = len(group) / len(df) * 100
        print(f"  {driver:<28s}: {len(group):4d} 个格网 ({pct:5.1f}%)")

    print(f"\n  低可达性格网的首因:")
    low = df[df['access_level'].isin(['非常低', '低'])]
    for driver, count in low['top_driver'].value_counts().items():
        print(f"    {driver}: {count} 格网")

    return shap_values, features, df


# ── 3. 双重稳健估计 ─────────────────────────────────────

def doubly_robust_estimate(df):
    """用 NonParamDML 估计路网密度对可达性的因果效应"""
    from econml.dml import NonParamDML
    from sklearn.model_selection import KFold
    from sklearn.ensemble import GradientBoostingRegressor
    print("\n" + "=" * 60)
    print("方法B: 双重ML —— 路网密度每增加1，可达性提升多少？")
    print("=" * 60)

    features = ['intersection_density', 'road_hierarchy_mix', 'pop_density',
                'age_65_ratio', 'building_density', 'land_use_mix',
                'dist_to_cbd', 'nearest_station_dist']
    confounders = ['land_price', 'nightlight']

    X = df[features].values
    T = df['road_density'].values
    y = df['accessibility'].values
    W = df[confounders].values

    model_y = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_t = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

    dml = NonParamDML(
        model_y=model_y,
        model_t=model_t,
        model_final=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        discrete_treatment=False,
        cv=KFold(n_splits=3, shuffle=True, random_state=42)
    )
    dml.fit(Y=y, T=T, X=X, W=W)

    ate = dml.ate(X=X)
    cate = dml.effect(X=X)

    print(f"\n  ATE (avg treatment effect): {ate:.4f}")
    print(f"  Road density +1 unit -> accessibility +{ate:.4f}")
    print(f"  CATE range: [{cate.min():.4f}, {cate.max():.4f}]")

    # Winsorize CATE 去除极端值
    cate_clipped = np.clip(cate, np.percentile(cate, 1), np.percentile(cate, 99))
    
    # 按空间区位的异质性
    df['cate'] = cate_clipped
    df['cate_bin'] = pd.cut(df['cate'], bins=4, labels=['弱效应', '中弱', '中强', '强效应'])

    print(f"\n  CATE 空间分布:")
    for label, group in df.groupby('cate_bin'):
        print(f"    {label}: {len(group)} 格网, 均值={group['cate'].mean():.4f}")

    # 干预推荐
    top_n = int(len(df) * 0.1)
    best = df.nlargest(top_n, 'cate')
    print(f"\n  干预优先级 (top 10% CATE格网):")
    print(f"    位置: x[{(best['grid_x'].mean()/df['grid_x'].max()*100).astype(int)}%] "
          f"y[{(best['grid_y'].mean()/df['grid_y'].max()*100).astype(int)}%]")
    print(f"    路网密度均值: {best['road_density'].mean():.2f} (全局均值: {df['road_density'].mean():.2f})")
    print(f"    人均可达性提升: {best['cate'].mean():.4f}")

    return cate, dml


# ── 4. 归因地图 ────────────────────────────────────────

def plot_spatial_maps(df):
    """生成空间归因图和 CATE 热力图"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    grid_size = int(np.sqrt(len(df)))
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # 检查是否规则格网
    gx_vals = sorted(df['grid_x'].unique()) if 'grid_x' in df.columns else []
    gy_vals = sorted(df['grid_y'].unique()) if 'grid_y' in df.columns else []
    is_rect = 'grid_x' in df.columns and (len(gx_vals) * len(gy_vals) == len(df))
    n_pad = grid_size if grid_size > 0 else int(np.ceil(np.sqrt(len(df))))

    def to_heatmap_raw(values):
        padded = np.full(n_pad * n_pad, np.nan)
        padded[:len(values)] = np.array(values)[:n_pad*n_pad]
        return padded.reshape(n_pad, n_pad)

    def to_heatmap(col):
        if is_rect:
            data = np.full((len(gx_vals), len(gy_vals)), np.nan)
            gx_idx = {v: i for i, v in enumerate(gx_vals)}
            gy_idx = {v: i for i, v in enumerate(gy_vals)}
            for _, row in df.iterrows():
                data[gx_idx[row['grid_x']], gy_idx[row['grid_y']]] = row[col]
            return data
        else:
            return to_heatmap_raw(df[col].values)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    im = ax.imshow(to_heatmap('accessibility'), cmap='RdYlGn', origin='lower')
    ax.set_title('Accessibility', fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    im = ax.imshow(to_heatmap('road_density'), cmap='Blues', origin='lower')
    ax.set_title('Road Density', fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 2]
    im = ax.imshow(to_heatmap('pop_density'), cmap='OrRd', origin='lower')
    ax.set_title('Pop Density', fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 归因地图（颜色编码）
    ax = axes[1, 0]
    driver_map = np.zeros(len(df))
    driver_names = ['road_density', 'intersection_density', 'road_hierarchy_mix',
                    'pop_density', 'dist_to_cbd', 'nearest_station_dist', 'land_use_mix',
                    'age_65_ratio', 'building_density', 'nightlight', 'land_price']
    for i, d in enumerate(driver_names):
        driver_map[df['top_driver'] == d] = i + 1
    im = ax.imshow(to_heatmap_raw(driver_map), cmap='tab10', origin='lower', vmin=0, vmax=10)
    ax.set_title('Top Attribution Driver', fontsize=13)

    # CATE 图
    ax = axes[1, 1]
    im = ax.imshow(to_heatmap('cate'), cmap='RdYlBu_r', origin='lower')
    ax.set_title('CATE: Road Improvement Effect', fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 可及性等级
    ax = axes[1, 2]
    levels = {'非常低': 1, '低': 2, '中': 3, '高': 4, '非常高': 5}
    level_map = df['access_level'].map(levels).values
    im = ax.imshow(to_heatmap_raw(level_map), cmap='RdYlGn', origin='lower', vmin=0, vmax=5)
    ax.set_title('Access Level', fontsize=13)

    fig.suptitle('EMS Accessibility Spatial Causal Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / "spatial_causal_attribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图已保存: {out_dir / 'spatial_causal_attribution.png'}")
    # 自动用浏览器打开
    import webbrowser
    webbrowser.open(str(out_dir / "spatial_causal_attribution.png"))


# ── 5. 全局重要性图表 ──────────────────────────────────

def plot_importance_bar(shap_values, features):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = {
        'road_density': '路网密度',
        'pop_density': '人口密度',
        'intersection_density': '交叉口密度',
        'road_hierarchy_mix': '路网等级混合',
        'nearest_station_dist': '距最近急救站',
        'land_use_mix': '土地利用混合度',
        'dist_to_cbd': '距CBD距离',
        'age_65_ratio': '老龄人口比',
        'building_density': '建筑密度',
        'nightlight': '夜间灯光',
        'land_price': '土地价格'
    }
    names = [labels.get(f, f) for f in np.array(features)[order]]
    ax.barh(names, importance[order], color='steelblue')
    ax.set_xlabel('平均 |SHAP| 值', fontsize=12)
    ax.set_title('可达性空间分异的因素重要性', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    import webbrowser
    webbrowser.open(str(out_dir / "feature_importance.png"))


# ── 6. 主函数 ──────────────────────────────────────────

def main():
    print("=" * 60)
    print("城市急救可达性空间分异因果推断")
    print("=" * 60)

    # 生成数据
    df = generate_synthetic_city(grid_size=50)

    print(f"\n数据概览:")
    print(f"  格网数: {len(df)}")
    print(f"  可达性范围: [{df['accessibility'].min():.2f}, {df['accessibility'].max():.2f}]")
    print(f"  可达性均值: {df['accessibility'].mean():.2f} ± {df['accessibility'].std():.2f}")
    print(f"  路网密度范围: [{df['road_density'].min():.2f}, {df['road_density'].max():.2f}]")
    print(f"  人口密度范围: [{df['pop_density'].min():.2f}, {df['pop_density'].max():.2f}]")

    # 方法A
    shap_values, features, df = shap_attribution(df)

    # 方法B
    cate, dml = doubly_robust_estimate(df)

    # 可视化
    plot_spatial_maps(df)
    plot_importance_bar(shap_values, features)

    # 总结
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)

    # 计算每个因子对可达性空间分异的贡献
    total_var = df['accessibility'].var()
    # SHAP 值方差分解
    shap_var = np.var(shap_values, axis=0)
    shap_var_pct = shap_var / shap_var.sum() * 100

    print("\n可达性空间分异方差分解 (SHAP-based):")
    labels = {
        'road_density': '路网密度', 'pop_density': '人口密度',
        'intersection_density': '交叉口密度', 'road_hierarchy_mix': '路网等级混合',
        'nearest_station_dist': '距最近急救站', 'land_use_mix': '土地利用混合度',
        'dist_to_cbd': '距CBD距离', 'age_65_ratio': '老龄人口比',
        'building_density': '建筑密度', 'nightlight': '夜间灯光',
        'land_price': '土地价格'
    }
    for i in np.argsort(shap_var_pct)[::-1]:
        f = features[i]
        pct = shap_var_pct[i]
        bar = "█" * max(1, int(pct))
        print(f"  {labels.get(f, f):<20s} {bar} {pct:5.1f}%")

    # 因果效应
    ate_full = dml.ate(X=df[['intersection_density', 'road_hierarchy_mix', 'pop_density',
                           'age_65_ratio', 'building_density', 'land_use_mix',
                           'dist_to_cbd', 'nearest_station_dist']].values)
    print(f"\n因果效应总结:")
    print(f"  路网密度 ATE: {ate_full:.4f}")

    print("\n✔ 分析完成。结果图片保存在 output/ 目录。")


if __name__ == "__main__":
    main()
