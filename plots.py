import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def create_visualizations(csv_path: str = "street_light_dataset.csv"):
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    print("\nLoading dataset...")
    dataset = pd.read_csv(csv_path)
    print(f"Loaded: {len(dataset)} samples")

    sns.set_style("whitegrid")
    time_axis = np.arange(len(dataset))
    week_hours, start, end = 7 * 24, 0, 7 * 24
    t_week = np.arange(week_hours)

    # 1. Feature Distributions
    print("\n1/12: Feature distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold")

    axes[0, 0].hist(dataset["hour"], bins=24, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Hour")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(dataset["day_of_week"], bins=7, edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Day of Week")
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].hist(dataset["temperature"], bins=30, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 2].set_title("Temperature (>0°C)")
    axes[0, 2].axvline(0, color="red", linestyle="--", label="Freezing")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].bar(["No Rain", "Rain"], dataset["rain"].value_counts().sort_index(), color=["skyblue", "darkblue"], alpha=0.7)
    axes[1, 0].set_title(f"Rain: {dataset['rain'].sum()}h")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].bar(["No Fog", "Fog"], dataset["fog"].value_counts().sort_index(), color=["lightgray", "gray"], alpha=0.7)
    axes[1, 1].set_title(f"Fog: {dataset['fog'].sum()}h (RARE)")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].hist(dataset["activity_index"], bins=30, edgecolor="black", alpha=0.7, color="darkred")
    axes[1, 2].set_title("Activity (More Gaussian)")
    axes[1, 2].axvline(0.12, color="red", linestyle="--", label="Min")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("01_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Activity by Hour
    print("2/12: Activity by hour...")
    fig, ax = plt.subplots(figsize=(14, 6))
    box_data = [dataset[dataset["hour"] == h]["activity_index"].values for h in range(24)]
    bp = ax.boxplot(box_data, tick_labels=range(24), patch_artist=True)
    [patch.set_facecolor("lightcoral") for patch in bp["boxes"]]
    ax.set_title("Activity by Hour", fontsize=14, fontweight="bold")
    ax.axhline(dataset["activity_index"].mean(), color="red", linestyle="--", label=f"Mean: {dataset['activity_index'].mean():.3f}")
    ax.axhline(0.12, color="blue", linestyle=":", label="Min")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("02_activity_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Weekday vs Weekend
    print("3/12: Weekday vs weekend...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    box_data_days = [dataset[dataset["day_of_week"] == d]["activity_index"].values for d in range(7)]
    bp = axes[0].boxplot(box_data_days, tick_labels=day_names, patch_artist=True)
    [patch.set_facecolor("lightblue" if i >= 5 else "lightgreen") for i, patch in enumerate(bp["boxes"])]
    axes[0].set_title("Activity by Day (Weekdays Busier)")
    axes[0].grid(True, alpha=0.3)

    day_means = dataset.groupby("day_of_week")["activity_index"].mean()
    axes[1].bar(day_names, day_means, color=["lightgreen"] * 5 + ["lightblue"] * 2, edgecolor="black", alpha=0.7)
    axes[1].axhline(dataset["activity_index"].mean(), color="red", linestyle="--", label="Overall Mean")
    axes[1].set_title("Mean Activity by Day")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("03_weekday_vs_weekend.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Weather Impact
    print("4/12: Weather impact...")
    dataset["weather"] = "Clear"
    dataset.loc[dataset["rain"] == 1, "weather"] = "Rain"
    dataset.loc[dataset["fog"] == 1, "weather"] = "Fog"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, cond in enumerate(["rain", "fog"]):
        groups = dataset.groupby(cond)["activity_index"].agg(["mean", "std", "count"])
        axes[idx].bar(
            [f"No {cond.title()}", cond.title()],
            groups["mean"],
            yerr=groups["std"],
            capsize=10,
            color=["skyblue", "darkblue"] if cond == "rain" else ["lightgray", "gray"],
            alpha=0.7,
        )
        axes[idx].set_title(f"{cond.title()} Impact ({'50%' if cond == 'rain' else '30%'} reduction)")
        axes[idx].grid(True, alpha=0.3, axis="y")

    w_groups = dataset.groupby("weather")["activity_index"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    axes[2].bar(range(len(w_groups)), w_groups["mean"], yerr=w_groups["std"], capsize=10, color=["gold", "gray", "darkblue"], alpha=0.7)
    axes[2].set_xticks(range(len(w_groups)))
    axes[2].set_xticklabels(w_groups.index)
    axes[2].set_title("Weather (Mutually Exclusive)")
    axes[2].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("04_weather_impact.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Hour vs Temp (Activity colors)
    print("5/12: Hour vs temperature...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(dataset["hour"], dataset["temperature"], c=dataset["activity_index"], cmap="coolwarm", alpha=0.5, s=15)
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Hour vs Temperature (colored by Activity)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[0], label="Activity")

    temp_bins = pd.cut(dataset["temperature"], bins=10)
    temp_act = dataset.groupby(temp_bins, observed=True)["activity_index"].agg(["mean", "std"])
    temp_centers = [interval.mid for interval in temp_act.index]
    axes[1].errorbar(temp_centers, temp_act["mean"], yerr=temp_act["std"], fmt="o-", capsize=5)
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Mean Activity")
    axes[1].set_title("Activity by Temperature")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("05_hour_temp_activity.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6. Correlation Heatmap
    print("6/12: Correlation heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    cols = ["hour", "day_of_week", "temperature", "rain", "fog", "activity_index", "visibility", "safety_risk"]
    corr = dataset[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0, square=True, linewidths=1, ax=ax)
    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7. Time Series
    print("7/12: Time series...")
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    tr = range(week_hours)
    axes[0].plot(tr, dataset["activity_index"].iloc[tr], "darkred", linewidth=1.5, label="Activity")
    rain_mask = dataset["rain"].iloc[tr] == 1
    fog_mask = dataset["fog"].iloc[tr] == 1
    axes[0].scatter(np.where(rain_mask)[0], dataset["activity_index"].iloc[tr][rain_mask], marker="v", s=80, c="blue", alpha=0.6, label="Rain")
    axes[0].scatter(np.where(fog_mask)[0], dataset["activity_index"].iloc[tr][fog_mask], marker="o", s=60, c="gray", alpha=0.6, label="Fog")
    axes[0].set_title("Activity with Weather (Week 1)", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    ax2 = axes[0].twinx()
    ax2.plot(tr, dataset["temperature"].iloc[tr], "orange", alpha=0.4, linewidth=1)
    ax2.set_ylabel("Temp (°C)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    sr = range(min(14 * 24, len(dataset)))
    norm_data = dataset[["temperature", "activity_index"]].iloc[sr].copy()
    norm_data["temperature"] = (norm_data["temperature"] - norm_data["temperature"].min()) / (
        norm_data["temperature"].max() - norm_data["temperature"].min()
    )
    axes[1].plot(sr, norm_data["activity_index"], "darkred", linewidth=1.5, label="Activity")
    axes[1].plot(sr, norm_data["temperature"], "orange", linewidth=1, alpha=0.7, label="Temp (scaled)")
    axes[1].set_title("Normalized Features (2 Weeks)", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    hourly = dataset.groupby("hour")["activity_index"].agg(["mean", "std"])
    axes[2].plot(hourly.index, hourly["mean"], "o-", linewidth=2, color="darkred", label="Mean")
    axes[2].fill_between(hourly.index, hourly["mean"] - hourly["std"], hourly["mean"] + hourly["std"], alpha=0.3, color="red")
    axes[2].set_title("Average Daily Pattern", fontweight="bold")
    axes[2].set_xlabel("Hour")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("07_time_series.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 8. Feature Interactions
    print("8/12: Feature interactions...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    rain_hour = dataset.groupby(["hour", "rain"])["activity_index"].mean().unstack()
    axes[0, 0].plot(rain_hour.index, rain_hour[0], "o-", label="No Rain")
    axes[0, 0].plot(rain_hour.index, rain_hour[1], "s-", label="Rain")
    axes[0, 0].set_title("Rain Effect by Hour")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    fog_hour = dataset.groupby(["hour", "fog"])["activity_index"].mean().unstack()
    axes[0, 1].plot(fog_hour.index, fog_hour[0], "o-", label="No Fog")
    axes[0, 1].plot(fog_hour.index, fog_hour[1], "s-", label="Fog")
    axes[0, 1].set_title("Fog Effect by Hour")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    dataset["is_weekend"] = dataset["day_of_week"] >= 5
    wknd_hour = dataset.groupby(["hour", "is_weekend"])["activity_index"].mean().unstack()
    axes[1, 0].plot(wknd_hour.index, wknd_hour[False], "o-", label="Weekday", linewidth=2)
    axes[1, 0].plot(wknd_hour.index, wknd_hour[True], "s-", label="Weekend", linewidth=2)
    axes[1, 0].set_title("Weekday vs Weekend by Hour (Weekdays Higher)", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    pivot = dataset.pivot_table(values="activity_index", index="hour", columns="day_of_week")
    sns.heatmap(pivot, cmap="YlOrRd", ax=axes[1, 1], cbar_kws={"label": "Activity"})
    axes[1, 1].set_title("Hour × Day Heatmap")
    axes[1, 1].set_xlabel("Day (0=Mon, 6=Sun)")

    plt.tight_layout()
    plt.savefig("08_feature_interactions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 9. Visibility
    print("9/12: Visibility...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    vis_counts = dataset["visibility"].value_counts().sort_index()
    axes[0, 0].bar(["Poor", "Moderate", "Good"], vis_counts, color=["red", "orange", "green"], alpha=0.7)
    axes[0, 0].set_title("Visibility Distribution")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    axes[0, 1].scatter(time_axis, dataset["visibility"], c=dataset["visibility"], cmap="RdYlGn", s=3, alpha=0.5)
    axes[0, 1].set_title("Visibility Timeline")
    axes[0, 1].grid(True, alpha=0.3)

    vis_hour = dataset.groupby(["hour", "visibility"]).size().unstack(fill_value=0)
    vis_pct = vis_hour.div(vis_hour.sum(axis=1), axis=0) * 100
    axes[1, 0].bar(vis_pct.index, vis_pct[2], label="Good", color="green", alpha=0.7)
    axes[1, 0].bar(vis_pct.index, vis_pct[1], bottom=vis_pct[2], label="Moderate", color="orange", alpha=0.7)
    axes[1, 0].bar(vis_pct.index, vis_pct[0], bottom=vis_pct[2] + vis_pct[1], label="Poor", color="red", alpha=0.7)
    axes[1, 0].set_title("Visibility by Hour")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].scatter(t_week, dataset["visibility"].iloc[start:end], c=dataset["visibility"].iloc[start:end], cmap="RdYlGn", s=30)
    axes[1, 1].set_title("Visibility (Week 1)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("09_visibility.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 10. Safety Risk
    print("10/12: Safety risk...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(dataset["safety_risk"], bins=40, edgecolor="black", alpha=0.7, color="crimson")
    axes[0, 0].axvline(dataset["safety_risk"].mean(), color="blue", linestyle="--", label=f"Mean: {dataset['safety_risk'].mean():.3f}")
    axes[0, 0].set_title("Safety Risk (Higher at Night)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time_axis, dataset["safety_risk"], "crimson", linewidth=0.6, alpha=0.8)
    axes[0, 1].set_title("Safety Risk Timeline")
    axes[0, 1].grid(True, alpha=0.3)

    risk_hour = dataset.groupby("hour")["safety_risk"].agg(["mean", "std"])
    axes[1, 0].plot(risk_hour.index, risk_hour["mean"], "o-", linewidth=2, color="crimson")
    axes[1, 0].fill_between(risk_hour.index, risk_hour["mean"] - risk_hour["std"], risk_hour["mean"] + risk_hour["std"], alpha=0.3, color="red")
    axes[1, 0].axvspan(20, 24, alpha=0.2, color="gray", label="Night")
    axes[1, 0].axvspan(0, 6, alpha=0.2, color="gray")
    axes[1, 0].set_title("Safety Risk by Hour")
    axes[1, 0].set_xticks(range(0, 24, 2))
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t_week, dataset["safety_risk"].iloc[start:end], "crimson", linewidth=1.2)
    axes[1, 1].set_title("Safety Risk (Week 1)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("10_safety_risk.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 11. Weather Timeline
    print("11/12: Weather timeline...")
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    rain_idx = np.where(dataset["rain"] == 1)[0]
    fog_idx = np.where(dataset["fog"] == 1)[0]
    clear_idx = np.where((dataset["rain"] == 0) & (dataset["fog"] == 0))[0]

    axes[0].scatter(rain_idx, np.ones(len(rain_idx)) * 2, marker="|", s=200, c="blue", alpha=0.6, label="Rain")
    axes[0].scatter(fog_idx, np.ones(len(fog_idx)), marker="|", s=200, c="gray", alpha=0.6, label="Fog")
    axes[0].scatter(clear_idx, np.zeros(len(clear_idx)), marker="|", s=100, c="gold", alpha=0.3, label="Clear")
    axes[0].set_title("Weather Timeline (Fog is RARE)", fontweight="bold")
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["Clear", "Fog", "Rain"])
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    rain_h = dataset.groupby("hour")["rain"].sum()
    fog_h = dataset.groupby("hour")["fog"].sum()
    clear_h = dataset.groupby("hour").apply(lambda x: ((x["rain"] == 0) & (x["fog"] == 0)).sum())

    x, width = np.arange(24), 0.25
    axes[1].bar(x - width, rain_h, width, label="Rain (any time)", color="blue", alpha=0.7)
    axes[1].bar(x, fog_h, width, label="Fog (morning/evening, rare)", color="gray", alpha=0.7)
    axes[1].bar(x + width, clear_h, width, label="Clear (most common)", color="gold", alpha=0.7)
    axes[1].set_title("Weather by Hour")
    axes[1].set_xticks(x)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    cats = ["Rain Only", "Fog Only", "Clear", "Both (Should be 0)"]
    counts = [
        ((dataset["rain"] == 1) & (dataset["fog"] == 0)).sum(),
        ((dataset["rain"] == 0) & (dataset["fog"] == 1)).sum(),
        ((dataset["rain"] == 0) & (dataset["fog"] == 0)).sum(),
        ((dataset["rain"] == 1) & (dataset["fog"] == 1)).sum(),
    ]
    axes[2].bar(cats, counts, color=["blue", "gray", "gold", "red"], alpha=0.7)
    [axes[2].text(i, counts[i] + 10, str(counts[i]), ha="center", fontweight="bold") for i in range(4)]
    axes[2].set_title("Weather Verification")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("11_weather_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 12. PCA
    print("12/12: PCA analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pca_features = ["hour", "day_of_week", "temperature", "rain", "fog", "activity_index", "visibility"]
    X = StandardScaler().fit_transform(dataset[pca_features])
    pca = PCA()
    X_pca = pca.fit_transform(X)

    axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7)
    axes[0, 0].plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), "ro-", linewidth=2)
    axes[0, 0].set_title("PCA Explained Variance\n(How much info each component captures)", fontsize=11)
    axes[0, 0].set_xlabel("Principal Component")
    axes[0, 0].set_ylabel("Variance Explained")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(
        0.5,
        0.95,
        f"PC1+PC2 explain {np.cumsum(pca.explained_variance_ratio_)[1] * 100:.1f}% of variance",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    sc = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=dataset["activity_index"], cmap="coolwarm", alpha=0.5, s=10)
    axes[0, 1].set_title("First 2 Principal Components\n(Each point is an hour, color = activity)", fontsize=11)
    axes[0, 1].set_xlabel("PC1 (Main pattern)")
    axes[0, 1].set_ylabel("PC2 (Secondary pattern)")
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[0, 1], label="Activity")

    pc1_contrib = pd.Series(pca.components_[0], index=pca_features)
    pc1_contrib.plot(kind="barh", ax=axes[1, 0], color="steelblue", alpha=0.7)
    axes[1, 0].set_title("PC1 Feature Contributions\n(What drives the main pattern?)", fontsize=11)
    axes[1, 0].set_xlabel("Contribution (positive = correlated)")
    axes[1, 0].grid(True, alpha=0.3)
    top_feature = pc1_contrib.abs().idxmax()
    axes[1, 0].text(
        0.02,
        0.98,
        f"Dominated by: {top_feature}",
        transform=axes[1, 0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    pc2_contrib = pd.Series(pca.components_[1], index=pca_features)
    pc2_contrib.plot(kind="barh", ax=axes[1, 1], color="coral", alpha=0.7)
    axes[1, 1].set_title("PC2 Feature Contributions\n(What drives the secondary pattern?)", fontsize=11)
    axes[1, 1].set_xlabel("Contribution")
    axes[1, 1].grid(True, alpha=0.3)
    top_feature2 = pc2_contrib.abs().idxmax()
    axes[1, 1].text(
        0.02,
        0.98,
        f"Dominated by: {top_feature2}",
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightsalmon", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig("12_pca_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print("✓ ALL 12 VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    for i in range(1, 13):
        print(f"  {i:02d}_*.png")

    print("\n" + "=" * 80)
    print("PCA ANALYSIS EXPLANATION")
    print("=" * 80)

    return dataset


if __name__ == "__main__":
    create_visualizations()
