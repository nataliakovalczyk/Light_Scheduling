import numpy as np
import pandas as pd


def generate_street_light_dataset(
    output_path: str = "street_light_dataset.csv",
    n_days: int = 60,
    hours_per_day: int = 24,
    seed: int = 42,
):
    np.random.seed(seed)

    N_DAYS, HOURS_PER_DAY = n_days, hours_per_day
    N_SAMPLES = N_DAYS * HOURS_PER_DAY

    print("=" * 80)
    print("STREET LIGHT DATASET GENERATION")
    print("=" * 80)
    print(f"\nGenerating {N_DAYS} days ({N_SAMPLES} hours)...")

    # Time
    print("\n1. Time index...")
    hours = np.tile(np.arange(HOURS_PER_DAY), N_DAYS)
    days = np.repeat(np.arange(N_DAYS), HOURS_PER_DAY)
    day_of_week = days % 7
    time_df = pd.DataFrame({"hour": hours, "day_of_week": day_of_week, "day": days})

    # Temperature (always > 0°C)
    print("2. Temperature...")
    base_temp = 12 + 6 * np.sin(2 * np.pi * (hours - 6) / 24)
    temperature = np.maximum(
        base_temp
        + np.repeat(np.random.normal(0, 2, N_DAYS), HOURS_PER_DAY)
        + np.random.normal(0, 1.0, N_SAMPLES),
        1.0,
    )

    # Fog (RARE, morning/evening only)
    print("3. Fog (rare)...")
    fog = np.zeros(N_SAMPLES, dtype=int)
    in_fog, fog_dur = False, 0
    for i in range(N_SAMPLES):
        if in_fog:
            fog[i], fog_dur = 1, fog_dur - 1
            if fog_dur <= 0:
                in_fog = False
        else:
            h = hours[i]
            prob = 0.08 if 5 <= h <= 8 else (0.05 if 21 <= h <= 23 else 0.0)  # REDUCED
            if np.random.random() < prob:
                in_fog, fog_dur, fog[i] = True, np.random.randint(3, 7), 1

    # Rain (any time, mutually exclusive with fog)
    print("4. Rain...")
    rain = np.zeros(N_SAMPLES, dtype=int)
    in_rain, rain_dur = False, 0
    for i in range(N_SAMPLES):
        if fog[i]:
            continue
        if in_rain:
            rain[i], rain_dur = 1, rain_dur - 1
            if rain_dur <= 0:
                in_rain = False
        else:
            h = hours[i]
            prob = 0.12 if 14 <= h <= 20 else 0.08
            if np.random.random() < prob:
                in_rain, rain_dur, rain[i] = True, np.random.randint(2, 11), 1

    weather_df = pd.DataFrame({"temperature": temperature, "rain": rain, "fog": fog})

    # Temperature-weather correlation (rain/fog lower temp)
    print("5. Adjusting temperature for weather correlation...")
    temperature_adjusted = temperature.copy()
    for i in range(N_SAMPLES):
        if rain[i]:
            temperature_adjusted[i] -= np.random.uniform(2, 5)  # Rain causes 2-5°C drop
        if fog[i]:
            temperature_adjusted[i] -= np.random.uniform(1, 3)  # Fog causes 1-3°C drop
        temperature_adjusted[i] = max(1.0, temperature_adjusted[i])

    temperature = temperature_adjusted
    weather_df["temperature"] = temperature

    # Activity (Gaussian centered at 0.55, WEEKDAYS BUSIER, with daily peaks)
    print("6. Activity (Gaussian peak at 0.5-0.6, weekdays busier, with daily variation)...")

    activity = np.zeros(N_SAMPLES)

    for i in range(N_SAMPLES):
        h, is_wknd = hours[i], day_of_week[i] >= 5
        d = days[i]

        base = np.random.normal(0.62, 0.15)
        time_adjustment = 1.0

        if not is_wknd:  # WEEKDAYS
            if 5 <= h < 7:
                time_adjustment = 0.85
            elif 7 <= h < 9:
                time_adjustment = 1.20
            elif 9 <= h < 12:
                time_adjustment = 1.30
            elif 12 <= h < 14:
                time_adjustment = 1.25
            elif 14 <= h < 17:
                time_adjustment = 1.10
            elif 17 <= h < 18:
                time_adjustment = 1.20
            elif 18 <= h < 21:
                time_adjustment = 1.05
            elif 21 <= h < 24:
                time_adjustment = 0.88
            else:
                time_adjustment = 0.70
        else:  # WEEKENDS
            if 5 <= h < 9:
                time_adjustment = 0.82
            elif 9 <= h < 12:
                time_adjustment = 0.98
            elif 12 <= h < 15:
                time_adjustment = 1.25
            elif 15 <= h < 18:
                time_adjustment = 1.10
            elif 18 <= h < 23:
                time_adjustment = 1.20
            elif 23 <= h < 24:
                time_adjustment = 0.92
            else:
                time_adjustment = 0.70

        base = base * time_adjustment

        # DAILY VARIATION
        daily_factor = 1.0 + np.sin(d * 0.3) * 0.08  # ±8%
        base = base * daily_factor

        # Weather effects
        if rain[i]:
            base *= 0.50
        if fog[i]:
            base *= 0.70
        if temperature[i] < 5:
            base *= 0.90
        elif temperature[i] > 25:
            base *= 0.93

        activity[i] = base

    activity = np.clip(activity, 0.20, 1.0)
    activity_df = pd.DataFrame({"activity_index": activity})

    # Visibility
    print("7. Visibility (Realistic model based on day cycle + weather)...")
    visibility_score = np.zeros(N_SAMPLES)

    for i in range(N_SAMPLES):
        h = hours[i]
        natural_light = 0.6 + 0.4 * np.cos((h - 14) * 2 * np.pi / 24)
        noise = np.random.normal(0, 0.05)
        current_vis = natural_light + noise

        if fog[i] == 1:
            current_vis -= np.random.uniform(0.5, 0.7)
        elif rain[i] == 1:
            current_vis -= np.random.uniform(0.2, 0.4)

        visibility_score[i] = np.clip(current_vis, 0.0, 1.0)

    visibility = np.zeros(N_SAMPLES, dtype=int)
    for i in range(N_SAMPLES):
        score = visibility_score[i]
        if score < 0.4:
            visibility[i] = 0
        elif score < 0.75:
            visibility[i] = 1
        else:
            visibility[i] = 2

    visibility_df = pd.DataFrame({"visibility": visibility})

    print("8. Safety risk (moderately higher at night)...")
    night_mult = np.ones(N_SAMPLES)
    for i in range(N_SAMPLES):
        h = hours[i]
        if 0 <= h <= 4:
            night_mult[i] = 1.35
        elif h >= 22 or h <= 6:
            night_mult[i] = 1.25
        elif h == 7 or h == 19 or h == 20:
            night_mult[i] = 1.15 if h == 20 else 1.10

    safety_risk = (0.5 * activity + 0.3 * (2 - visibility) / 2 + 0.2) * night_mult
    safety_df = pd.DataFrame({"lighting_level": np.ones(N_SAMPLES), "safety_risk": safety_risk})

    # Combine
    print("9. Combining...")
    dataset = pd.concat([time_df, weather_df, activity_df, visibility_df, safety_df], axis=1)

    # Save
    print("\n" + "=" * 80)
    print("SAVING")
    print("=" * 80)
    dataset.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"Size: {len(dataset)} rows × {len(dataset.columns)} columns")

    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"\n{'Feature':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    for col in [
        "hour",
        "day_of_week",
        "temperature",
        "rain",
        "fog",
        "activity_index",
        "visibility",
        "safety_risk",
    ]:
        print(
            f"{col:<20} {dataset[col].mean():<10.2f} {dataset[col].std():<10.2f} "
            f"{dataset[col].min():<10.2f} {dataset[col].max():<10.2f}"
        )

    weekday_act = activity[day_of_week < 5]
    weekend_act = activity[day_of_week >= 5]
    print(f"\nWeekday activity:  {weekday_act.mean():.4f}")
    print(f"Weekend activity:  {weekend_act.mean():.4f}")
    print(f"Difference:        {weekday_act.mean() - weekend_act.mean():.4f}")
    print(f"                   ({((weekday_act.mean() / weekend_act.mean() - 1) * 100):.1f}% higher on weekdays) ✓")

    print(f"\nRain: {rain.sum()}h ({100 * rain.sum() / N_SAMPLES:.1f}%)")
    print(f"Fog:  {fog.sum()}h ({100 * fog.sum() / N_SAMPLES:.1f}%) [RARE] ✓")
    print(f"Rain & Fog overlap: {((rain == 1) & (fog == 1)).sum()} ✓")
    print(f"Temp > 0°C: {(temperature > 0).all()} ✓")
    print(f"Activity never zero: {(activity >= 0.12).all()} ✓")

    rain_red = (1 - dataset[dataset["rain"] == 1]["activity_index"].mean()
                / dataset[dataset["rain"] == 0]["activity_index"].mean()) * 100
    fog_red = (1 - dataset[dataset["fog"] == 1]["activity_index"].mean()
               / dataset[dataset["fog"] == 0]["activity_index"].mean()) * 100
    print(f"\nRain impact: {rain_red:.1f}% reduction")
    print(f"Fog impact:  {fog_red:.1f}% reduction")

    # Temperature-weather correlation
    print("\n" + "-" * 80)
    print("TEMPERATURE-WEATHER CORRELATION")
    print("-" * 80)
    temp_no_rain = dataset[dataset["rain"] == 0]["temperature"].mean()
    temp_rain = dataset[dataset["rain"] == 1]["temperature"].mean()
    temp_no_fog = dataset[dataset["fog"] == 0]["temperature"].mean()
    temp_fog = dataset[dataset["fog"] == 1]["temperature"].mean()
    print(f"Temperature without rain: {temp_no_rain:.2f}°C")
    print(f"Temperature with rain:    {temp_rain:.2f}°C (drop: {temp_no_rain - temp_rain:.2f}°C) ✓")
    print(f"Temperature without fog:  {temp_no_fog:.2f}°C")
    print(f"Temperature with fog:     {temp_fog:.2f}°C (drop: {temp_no_fog - temp_fog:.2f}°C) ✓")

    day_mask = (dataset["hour"] >= 7) & (dataset["hour"] <= 19)
    print(f"\nNight safety: {dataset[~day_mask]['safety_risk'].mean():.3f}")
    print(f"Day safety:   {dataset[day_mask]['safety_risk'].mean():.3f}")
    print(f"Multiplier:   {dataset[~day_mask]['safety_risk'].mean() / dataset[day_mask]['safety_risk'].mean():.2f}x ✓")

    return dataset


if __name__ == "__main__":
    generate_street_light_dataset()
