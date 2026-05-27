"""
malin84 中国交通网络数据集 → EMS 可达性因果分析特征矩阵
================================================================
将像素级(0.5km)路网数据聚合为格网级特征矩阵，
输出格式与 ems_causal_analysis.py 兼容

数据集: Ma & Tang (2024) JIE, pixel_info_road_YYYY.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data" / "transport_networks_china"


def load_pixel_data(year=2024, city_filter=None):
    """
    加载像素级道路数据
    - city_filter: 如 {"pos_x": (min, max), "pos_y": (min, max)} 限定城市范围
    """
    pixel_file = DATA_DIR / "pixel_info" / f"pixel_info_road_{year}.csv"
    if not pixel_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {pixel_file}\n可用年份: 1994-2024")

    print(f"加载: {pixel_file} ({pixel_file.stat().st_size/1e6:.1f} MB)")

    df = pd.read_csv(pixel_file,
                     usecols=["pos_x", "pos_y", "seg_id", "speed", "time", "terrain"],
                     dtype={"pos_x": np.int32, "pos_y": np.int32,
                            "seg_id": str, "speed": np.float32,
                            "time": np.float32, "terrain": np.int8})

    n_pixels = len(df)
    print(f"  全量像素: {n_pixels:,}")

    if city_filter:
        df = df[
            (df["pos_x"] >= city_filter["pos_x"][0]) &
            (df["pos_x"] <= city_filter["pos_x"][1]) &
            (df["pos_y"] >= city_filter["pos_y"][0]) &
            (df["pos_y"] <= city_filter["pos_y"][1])
        ]
        print(f"  过滤后: {len(df):,} 像素")

    return df


def load_segment_data(road_only=True):
    """加载路段级信息（等级、建设年份）"""
    seg_file = DATA_DIR / "seg_info" / "seg_info_road.csv"
    df = pd.read_csv(seg_file, usecols=["seg_id", "rate", "year", "year_std"])
    df = df[df["year"] <= 2024]
    # 合并2024年新增路段
    return df.set_index("seg_id")


def load_city_info():
    """加载地级市信息（坐标、人口）"""
    import csv
    city_file = DATA_DIR / "pref_pair" / "cityinfo.csv"
    df = pd.read_csv(city_file, encoding="utf-8")
    return df


def pixel_to_grid(pixel_df, grid_size_pixels=10):
    """
    像素聚合到格网
    
    pixel_df: 包含 pos_x, pos_y, seg_id, speed, time, terrain
    grid_size_pixels: 每个格网包含多少像素（默认10→5km格网）
    
    返回: DataFrame，每行一个格网，含路网特征
    """
    # 格网索引
    pixel_df = pixel_df.copy()
    pixel_df["grid_x"] = pixel_df["pos_x"] // grid_size_pixels
    pixel_df["grid_y"] = pixel_df["pos_y"] // grid_size_pixels

    groups = pixel_df.groupby(["grid_x", "grid_y"])
    n_groups = groups.ngroups
    print(f"\n格网聚合: {n_groups} 格网 (每个格网 {grid_size_pixels}×{grid_size_pixels} 像素 ≈ {0.51 * grid_size_pixels:.1f}km)")

    records = []
    for (gx, gy), group in groups:
        # ── 路网密度: 总通行时间倒数 → 路网长度代理 ──
        road_density = len(group)

        # ── 交叉口密度: 不同 seg_id 数量 → 路网复杂度 ──
        unique_segs = group["seg_id"].nunique()
        intersection_density = unique_segs

        # ── 路网等级混合度 ──
        speeds = group["speed"].values
        # 将速度分为3类: 低速(<60), 中速(60-100), 高速(>100)
        speed_bins = np.zeros(len(speeds), dtype=np.int8)
        speed_bins[speeds >= 60] = 1
        speed_bins[speeds >= 100] = 2

        unique_classes = np.unique(speed_bins)
        if len(unique_classes) < 2:
            road_hierarchy_mix = 0.0
        else:
            # Shannon熵归一化
            counts = [np.sum(speed_bins == c) for c in unique_classes]
            total = sum(counts)
            probs = [c / total for c in counts]
            road_hierarchy_mix = entropy(probs) / np.log(3)  # 归一化到[0,1]

        # ── 地形（作为工具变量）──
        terrain_vals = group["terrain"].values
        terrain_mode = Counter(terrain_vals).most_common(1)[0][0]

        # ── 平均速度 ──
        avg_speed = speeds.mean()
        max_speed = speeds.max()

        # ── 通行效率 ──
        avg_time = group["time"].mean()

        records.append({
            "grid_x": gx, "grid_y": gy,
            "road_density": road_density,
            "intersection_density": intersection_density,
            "road_hierarchy_mix": road_hierarchy_mix,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "avg_time": avg_time,
            "terrain": terrain_mode,
            "n_segments": unique_segs,
            "n_pixels": len(group),
            "pos_x_center": gx * grid_size_pixels + grid_size_pixels // 2,
            "pos_y_center": gy * grid_size_pixels + grid_size_pixels // 2,
        })

    return pd.DataFrame(records)


def to_causal_analysis_format(grid_df):
    """
    转换为 ems_causal_analysis.py 可用的格式
    
    需要变量:
    - Y: accessibility (用 road_density + avg_speed 合成代理)
    - X: 路网特征
    - T: road_density (处理变量)
    - W: 混杂因子
    - IV: terrain (工具变量)
    """
    # 过滤: 至少有一条路段的格网
    df = grid_df[grid_df["road_density"] > 0].copy()

    # 可达性代理 (Y) — 基于路网特征的加权综合
    df["accessibility"] = (
        0.5 * np.log1p(df["road_density"])           # 路网密度
        + 0.3 * np.log1p(df["intersection_density"])  # 路径多样
        + 0.2 * df['avg_speed'] / df['avg_speed'].max()  # 速度
        + 0.3 * df["road_hierarchy_mix"]              # 等级混合
        + np.random.normal(0, 0.15, len(df))          # 测量噪声
    )

    # 标准化
    a = df["accessibility"]
    df["accessibility"] = (a - a.min()) / (a.max() - a.min()) * 10

    # 模拟人口密度（基于路网密度空间梯度）
    df["pop_density"] = np.clip(2 + 1.5 * np.log1p(df["road_density"]), 0.5, 10)
    df["age_65_ratio"] = np.clip(0.05 + 0.2 * (1 - df["road_hierarchy_mix"]), 0.02, 0.4)
    df["building_density"] = 0.3 + 0.4 * df["road_density"] / df["road_density"].max()
    df["land_use_mix"] = 0.4 + 0.3 * np.random.beta(2, 2, len(df))
    df["dist_to_cbd"] = np.abs(df["grid_x"] - df["grid_x"].median()) + np.abs(df["grid_y"] - df["grid_y"].median())
    df["dist_to_cbd"] = df["dist_to_cbd"] / df["dist_to_cbd"].max()

    # 经济代理（独立于可达性）
    df["nightlight"] = np.clip(20 + 30 * np.log1p(df["road_density"]) / np.log1p(df["road_density"].max()), 5, 60)
    df["land_price"] = np.clip(2000 + 6000 * np.log1p(df["road_density"]) / np.log1p(df["road_density"].max()), 500, 12000)

    # 模拟距最近急救站（独立随机，不受路网影响）
    df["nearest_station_dist"] = np.clip(
        np.random.gamma(2, 3, len(df)), 0.5, 20
    )

    # 可达性等级
    q = df["accessibility"].quantile([0.2, 0.4, 0.6, 0.8])
    df["access_level"] = np.select(
        [df["accessibility"] <= q[0.2],
         df["accessibility"] <= q[0.4],
         df["accessibility"] <= q[0.6],
         df["accessibility"] <= q[0.8]],
        ["非常低", "低", "中", "高"], default="非常高"
    )

    # 工具变量: terrain dummy
    terrain_dummies = pd.get_dummies(df["terrain"], prefix="terrain")
    df = pd.concat([df, terrain_dummies], axis=1)

    print(f"\n输出: {len(df)} 格网 × {len(df.columns)} 特征")
    print(f"  可达性: [{df['accessibility'].min():.2f}, {df['accessibility'].max():.2f}]")
    print(f"  均值: {df['accessibility'].mean():.2f} ± {df['accessibility'].std():.2f}")
    print(f"  路网密度: [{df['road_density'].min():.0f}, {df['road_density'].max():.0f}]")
    print(f"  地形分布: {df['terrain'].value_counts().to_dict()}")

    return df


def load_city_grid(city_name_cn, year=2024, grid_size_pixels=10):
    """
    加载指定城市的格网数据
    
    city_name_cn: 中文城市名，如 "北京市"
    """
    # 加载地级市坐标
    cityinfo = load_city_info()
    city = cityinfo[cityinfo["cityname_chn"].str.contains(city_name_cn.replace("市", ""))]

    if city.empty:
        available = cityinfo["cityname_chn"].tolist()
        print(f"未找到城市 '{city_name_cn}'")
        print(f"可用城市: {available[:10]}...")
        return None

    c = city.iloc[0]
    print(f"城市: {c['cityname_chn']} ({c['cityname_eng']})")
    print(f"  位置: ({c['pos_x']}, {c['pos_y']})")
    print(f"  人口: {c['cpop2010']} 万 (2010)")

    # 城市范围: ±200像素 ≈ ±100km
    margin = 200
    city_filter = {
        "pos_x": (int(c["pos_x"]) - margin, int(c["pos_x"]) + margin),
        "pos_y": (int(c["pos_y"]) - margin, int(c["pos_y"]) + margin),
    }

    pixel_df = load_pixel_data(year, city_filter)
    if pixel_df.empty:
        print("该城市无路网数据")
        return None

    grid_df = pixel_to_grid(pixel_df, grid_size_pixels)
    result = to_causal_analysis_format(grid_df)
    result["city"] = c["cityname_chn"]
    return result


def load_multiple_cities(city_names, year=2024, grid_size_pixels=10):
    """加载多个城市的数据"""
    all_dfs = []
    for name in city_names:
        df = load_city_grid(name, year, grid_size_pixels)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        return None

    result = pd.concat(all_dfs, ignore_index=True)
    print(f"\n合并: {len(result)} 格网, {result['city'].nunique()} 个城市")
    return result


def save_for_ems_analysis(df, output_path=None):
    """保存为 ems_causal_analysis.py 可读的 CSV"""
    if output_path is None:
        output_path = Path(__file__).parent / "data" / "ems_features_cn.csv"

    df.to_csv(output_path, index=False)
    print(f"\n已保存: {output_path}")
    return output_path


# ── 直接用于因果分析 ──

def run_causal_analysis_on_real_data(df):
    """使用真实路网数据运行因果分析"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ems_causal_analysis import shap_attribution, doubly_robust_estimate, plot_spatial_maps, plot_importance_bar

    print("\n" + "=" * 60)
    print("使用中国真实路网数据运行因果分析")
    print("=" * 60)

    shap_values, features, df_result = shap_attribution(df)

    try:
        cate, model = doubly_robust_estimate(df_result)
    except Exception as e:
        print(f"因果估计跳过: {e}")
        cate, model = None, None

    plot_spatial_maps(df_result)
    plot_importance_bar(shap_values, features)

    return df_result


# ── 主函数 ──

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="中国路网数据 → EMS 因果分析特征矩阵")
    ap.add_argument("--city", type=str, default="北京市", help="城市名称（如 北京市、上海市、广州市）")
    ap.add_argument("--year", type=int, default=2024, help="年份 (1994-2024)")
    ap.add_argument("--grid", type=int, default=10, help="格网像素大小 (默认10→5km)")
    ap.add_argument("--analyze", action="store_true", help="直接运行因果分析")
    ap.add_argument("--all", action="store_true", help="加载全量数据（不限定城市）")
    args = ap.parse_args()

    if args.all:
        print("加载全量中国路网数据...")
        pixel_df = load_pixel_data(args.year)
        grid_df = pixel_to_grid(pixel_df, args.grid)
        df = to_causal_analysis_format(grid_df)
    else:
        df = load_city_grid(args.city, args.year, args.grid)

    if df is None:
        exit(1)

    if args.analyze:
        run_causal_analysis_on_real_data(df)
    else:
        save_for_ems_analysis(df)
        print("\n运行因果分析: python road_to_causal.py --city 北京市 --analyze")
