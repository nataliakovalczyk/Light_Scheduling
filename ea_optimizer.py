# ea_optimizer.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed


# -----------------------------
# Shared / baseline EA settings
# -----------------------------
ENERGY_COST = {0: 0.0, 1: 0.5, 2: 1.0}

LAMBDA_ENERGY = 0.4
LAMBDA_SAFETY = 0.5
DARKNESS_RISK = 1.2


def night_penalty(hour, level):
    if (hour >= 22 or hour <= 6) and level == 0:
        return 2.5
    return 0.0


def build_hourly_profiles(dataset: pd.DataFrame):
    """
    Computes mean activity + visibility per hour exactly like your Colab.
    Returns:
      activity_hourly (np.ndarray shape [24]),
      visibility_hourly (np.ndarray shape [24]),
      visibility_norm (np.ndarray shape [24])
    """
    hourly_data = dataset.groupby("hour").mean(numeric_only=True)
    activity_hourly = hourly_data["activity_index"].values
    visibility_hourly = hourly_data["visibility"].values
    visibility_norm = visibility_hourly / 2.0
    return activity_hourly, visibility_hourly, visibility_norm


def build_baseline_schedule(hours: int = 24):
    baseline_schedule = np.zeros(hours, dtype=int)
    for h in range(hours):
        if 18 <= h <= 23 or 0 <= h <= 5:
            baseline_schedule[h] = 2  # full
        else:
            baseline_schedule[h] = 1  # dim
    return baseline_schedule


# -----------------------------
# Standard EA (energy + safety)
# -----------------------------
def fitness(schedule, activity_hourly, visibility_hourly, hours: int = 24):
    """
    Calculates the fitting of the schedule, the less the better.
    Normalised to ~[0, 1].
    """
    energy_sum = sum(ENERGY_COST[level] for level in schedule)
    max_possible_energy = hours * 1.0
    energy_norm = energy_sum / max_possible_energy

    safety_penalty = 0.0
    for h in range(hours):
        base_risk = activity_hourly[h] * (2 - visibility_hourly[h])
        dark_risk = DARKNESS_RISK * (2 - schedule[h])
        np_pen = night_penalty(h, schedule[h])
        safety_penalty += base_risk + dark_risk + np_pen

    safety_norm = safety_penalty / (hours * 3.0)

    return LAMBDA_ENERGY * energy_norm + LAMBDA_SAFETY * safety_norm


def mutate(schedule, mutation_rate=0.1):
    new_schedule = schedule.copy()
    for h in range(len(schedule)):
        if np.random.random() < mutation_rate:
            new_schedule[h] = np.random.randint(0, 3)
    return new_schedule


def crossover(parent1, parent2):
    hours = len(parent1)
    point = np.random.randint(1, hours - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def run_standard_ea(
    activity_hourly,
    visibility_hourly,
    baseline_schedule,
    pop_size: int = 50,
    n_generations: int = 100,
    mutation_rate: float = 0.1,
    elitism_count: int = 2,
):
    hours = len(baseline_schedule)
    population = [np.zeros(hours, dtype=int) for _ in range(pop_size)]
    best_fitness_history = []
    best_schedule = None
    best_fitness_val = float("inf")

    print(f"Starting EA optimization ({n_generations} generations)...")

    for gen in range(n_generations):
        fitnesses = np.array(
            [fitness(ind, activity_hourly, visibility_hourly, hours=hours) for ind in population]
        )

        current_best_idx = np.argmin(fitnesses)
        current_best_fit = fitnesses[current_best_idx]

        if current_best_fit < best_fitness_val:
            best_fitness_val = current_best_fit
            best_schedule = population[current_best_idx].copy()

        best_fitness_history.append(current_best_fit)

        sorted_indices = np.argsort(fitnesses)
        survivors = [population[i] for i in sorted_indices[: pop_size // 2]]

        new_population = []

        # elitism
        for i in range(elitism_count):
            new_population.append(population[sorted_indices[i]])

        while len(new_population) < pop_size:
            p1 = survivors[np.random.randint(len(survivors))]
            p2 = survivors[np.random.randint(len(survivors))]

            c1, c2 = crossover(p1, p2)

            new_population.append(mutate(c1, mutation_rate))
            if len(new_population) < pop_size:
                new_population.append(mutate(c2, mutation_rate))

        population = new_population

    print("Optimization finished.")
    print(f"Baseline Fitness: {fitness(baseline_schedule, activity_hourly, visibility_hourly, hours=hours):.4f}")
    print(f"Best EA Fitness:  {best_fitness_val:.4f}")

    return best_schedule, best_fitness_val, best_fitness_history


def plot_standard_ea_results(best_fitness_history, baseline_schedule, best_schedule, hours: int = 24):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(best_fitness_history, color="green")
    plt.title("EA Convergence (Fitness over Generations)")
    plt.xlabel("Generation")
    plt.ylabel("Loss (Min Fitness)")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.step(range(hours), baseline_schedule, where="mid", label="Baseline", linestyle="--")
    plt.step(range(hours), best_schedule, where="mid", label="EA Optimized", linewidth=2)
    plt.title("Optimized Schedule (Average Day)")
    plt.xlabel("Hour")
    plt.ylabel("Light Level (0=Off, 1=Dim, 2=Full)")
    plt.yticks([0, 1, 2], ["OFF", "DIM", "FULL"])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Fuzzy logic helpers
# -----------------------------
def trimf(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))


def activity_low(x):
    return trimf(x, -0.1, 0.2, 0.6)


def activity_medium(x):
    return trimf(x, 0.2, 0.6, 1.0)


def activity_high(x):
    return trimf(x, 0.6, 1.0, 1.1)


def visibility_poor(x):
    return trimf(x, -0.1, 0.0, 0.7)


def visibility_moderate(x):
    return trimf(x, 0.2, 0.5, 0.8)


def visibility_good(x):
    return trimf(x, 0.5, 1.0, 1.1)


def plot_membership_functions():
    x_axis = np.linspace(0, 1, 200)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x_axis, activity_low(x_axis), label="Low", color="green")
    plt.plot(x_axis, activity_medium(x_axis), label="Medium", color="orange")
    plt.plot(x_axis, activity_high(x_axis), label="High", color="red")
    plt.title("Membership Functions: Activity")
    plt.xlabel("Normalized Activity Index")
    plt.ylabel("Degree of Membership")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x_axis, visibility_poor(x_axis), label="Poor", color="red")
    plt.plot(x_axis, visibility_moderate(x_axis), label="Moderate", color="orange")
    plt.plot(x_axis, visibility_good(x_axis), label="Good", color="green")
    plt.title("Membership Functions: Visibility")
    plt.xlabel("Normalized Visibility (0=Bad, 1=Good)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_fuzzy_priority(activity, visibility):
    act_lo = trimf(activity, -0.1, 0.0, 0.5)
    act_md = trimf(activity, 0.3, 0.6, 0.9)
    act_hi = trimf(activity, 0.7, 1.0, 1.1)

    vis_po = trimf(visibility, -0.1, 0.0, 0.6)
    vis_md = trimf(visibility, 0.4, 0.7, 1.0)
    vis_gd = trimf(visibility, 0.8, 1.0, 1.1)

    fire_high = np.maximum(vis_po, act_hi)
    fire_medium = np.maximum(vis_md, act_md)
    fire_low = np.minimum(act_lo, vis_gd)

    numerator = (fire_high * 1.0) + (fire_medium * 0.5) + (fire_low * 0.1)
    denominator = fire_high + fire_medium + fire_low

    if denominator == 0:
        raw_output = 0.5
    else:
        raw_output = numerator / denominator

    if vis_po > 0.8:
        raw_output = max(raw_output, 0.5)

    return raw_output


def build_target_priority_curve(activity_hourly, visibility_norm, hours: int = 24):
    target_priority_curve = np.zeros(hours)
    for h in range(hours):
        target_priority_curve[h] = get_fuzzy_priority(activity_hourly[h], visibility_norm[h])
    return target_priority_curve


def fitness_fuzzy(schedule, target_priority_curve, hours: int = 24):
    energy_sum = sum(ENERGY_COST[level] for level in schedule)
    energy_norm = energy_sum / (hours * 1.0)

    deviation_penalty = 0.0
    for h in range(hours):
        light_norm = schedule[h] / 2.0
        target = target_priority_curve[h]
        deviation_penalty += (light_norm - target) ** 2
        deviation_penalty += night_penalty(h, schedule[h])

    deviation_norm = deviation_penalty / hours

    return 0.2 * energy_norm + 0.7 * deviation_norm


def run_fuzzy_ea(
    target_priority_curve,
    pop_size: int = 50,
    n_generations: int = 100,
    mutation_rate: float = 0.1,
    elitism_count: int = 2,
):
    hours = len(target_priority_curve)
    population = [np.random.randint(0, 3, size=hours) for _ in range(pop_size)]

    best_fuzzy_history = []
    best_fuzzy_schedule = None
    best_fuzzy_val = float("inf")

    for gen in range(n_generations):
        fitnesses = np.array([fitness_fuzzy(ind, target_priority_curve, hours=hours) for ind in population])

        current_best_idx = np.argmin(fitnesses)
        if fitnesses[current_best_idx] < best_fuzzy_val:
            best_fuzzy_val = fitnesses[current_best_idx]
            best_fuzzy_schedule = population[current_best_idx].copy()

        best_fuzzy_history.append(fitnesses[current_best_idx])

        survivors = [population[i] for i in np.argsort(fitnesses)[: pop_size // 2]]
        new_population = []
        new_population.extend([population[i] for i in np.argsort(fitnesses)[:elitism_count]])

        while len(new_population) < pop_size:
            p1 = survivors[np.random.randint(len(survivors))]
            p2 = survivors[np.random.randint(len(survivors))]
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1, mutation_rate))
            if len(new_population) < pop_size:
                new_population.append(mutate(c2, mutation_rate))

        population = new_population

    return best_fuzzy_schedule, best_fuzzy_val, best_fuzzy_history


def plot_fuzzy_results(best_fuzzy_history, best_fuzzy_schedule, target_priority_curve, hours: int = 24):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(best_fuzzy_history, label="Fuzzy EA Loss", color="purple")
    plt.title("Convergence: Fuzzy Optimization")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(hours), target_priority_curve * 2, "r--", label="Fuzzy Target (Ideal)", linewidth=2)
    plt.step(range(hours), best_fuzzy_schedule, where="mid", label="Optimized Schedule", color="blue", linewidth=3)
    plt.fill_between(range(hours), 0, target_priority_curve * 2, color="red", alpha=0.1)
    plt.title("Execution vs Intention (Schedule vs Fuzzy Target)")
    plt.xlabel("Hour")
    plt.ylabel("Light Level (0, 1, 2)")
    plt.yticks([0, 1, 2], ["OFF", "DIM", "FULL"])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Meta-optimization section (10)
# -----------------------------
def run_meta_optimization(
    activity_hourly,
    visibility_norm,
    hours: int = 24,
    chrom_dim: int = 2,
    outer_pop: int = 50,
    outer_gen: int = 30,
    outer_mut_rate: float = 0.15,
    activity_floor: float = 0.20,
    buffer: float = 0.05,
    seed: int = 42,
):
    rng_outer = np.random.default_rng(seed)

    def random_chromosome():
        return rng_outer.random(chrom_dim)

    def mutate_chromosome(chrom):
        c = chrom.copy()
        for i in range(chrom_dim):
            if rng_outer.random() < outer_mut_rate:
                c[i] = rng_outer.random()
        return c

    def unpack_chromosome(chrom):
        start_search = activity_floor + buffer
        span = 0.40

        aA = start_search + (chrom[0] * span)
        act_params = [aA, aA + 0.25, 1.2]

        end_poor = chrom[1] * 0.8
        vis_params = [-0.1, 0.0, end_poor]

        return act_params, vis_params

    def fuzzy_priority_tuned_local(activity, visibility, act_params, vis_params):
        aA, bA, cA = act_params
        aV, bV, cV = vis_params

        act_hi = trimf(activity, aA, bA, cA)
        vis_po = trimf(visibility, aV, bV, cV)

        return np.maximum(act_hi, vis_po)

    def optimize_schedule_strict(chrom, seed=None):
        act_params, vis_params = unpack_chromosome(chrom)

        local_target = np.zeros(hours)
        for h in range(hours):
            local_target[h] = fuzzy_priority_tuned_local(
                activity_hourly[h], visibility_norm[h], act_params, vis_params
            )

        if np.sum(local_target) < 0.5:
            return 99999.0

        ideal_schedule = np.round(local_target * 2.0).astype(int)

        current_cost = np.sum([ENERGY_COST[l] for l in ideal_schedule])
        energy_score = current_cost / (ENERGY_COST[2] * hours)

        risk_score = 0.0
        for h in range(hours):
            required = max(activity_hourly[h], (1.0 - visibility_norm[h]))
            provided = ideal_schedule[h] / 2.0
            if provided < required:
                risk_score += (required - provided) ** 2
        risk_score = risk_score / hours

        total_fitness = (energy_score * 2.0) + (risk_score * 12.0)
        return total_fitness

    def outer_fitness(chrom, eval_seed=0):
        return optimize_schedule_strict(chrom, seed=eval_seed)

    outer_population = [random_chromosome() for _ in range(outer_pop)]
    best_chrom = None
    best_fit = np.inf
    best_outer_history = []

    print(f"Starting Optimization (Floor: {activity_floor}, Buffer: {buffer})...")

    for gen in range(outer_gen):
        fits = Parallel(n_jobs=-1)(delayed(outer_fitness)(c) for c in outer_population)
        fits = np.array(fits)

        gen_best_idx = np.argmin(fits)
        gen_best_fit = fits[gen_best_idx]
        best_outer_history.append(gen_best_fit)

        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_chrom = outer_population[gen_best_idx].copy()

        if gen % 5 == 0:
            print(f"Gen {gen+1:02d}/{outer_gen} | Best Fit: {gen_best_fit:.4f}")

        idx = np.argsort(fits)[: outer_pop // 2]
        survivors = [outer_population[i] for i in idx]
        new_pop = survivors.copy()
        while len(new_pop) < outer_pop:
            p = survivors[np.random.randint(len(survivors))]
            new_pop.append(mutate_chromosome(p))
        outer_population = new_pop

    tuned_act, tuned_pri = unpack_chromosome(best_chrom)
    print(f"Tuned Thresholds -> Act High Starts: {tuned_act[0]:.2f}, Vis Poor Ends: {tuned_pri[2]:.2f}")

    test_target = [
        fuzzy_priority_tuned_local(activity_hourly[h], visibility_norm[h], tuned_act, tuned_pri)
        for h in range(hours)
    ]
    test_sched = np.round(np.array(test_target) * 2.0).astype(int)
    final_energy = np.sum([ENERGY_COST[l] for l in test_sched]) / (hours * 1.0) * 100
    print(f"Estimated Energy Usage: {final_energy:.2f}% ")

    # Plot section 10
    x_axis = np.linspace(0, 1, 200)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_axis, trimf(x_axis, 0.6, 1.0, 1.2), "k--", label="Handcrafted (Old)", alpha=0.5)
    plt.plot(x_axis, trimf(x_axis, *tuned_act), "r-", label="AI Tuned (Optimized)", linewidth=2)

    plt.axvline(activity_floor, color="blue", linestyle=":", label=f"Data Floor ({activity_floor})")
    plt.title("Evolution of 'High Activity' Concept")
    plt.xlabel("Activity Index")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(best_outer_history, marker="o", color="purple")
    plt.title("Meta-Optimization Convergence")
    plt.xlabel("Outer Generation")
    plt.ylabel("Best Fitness Achieved")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return tuned_act, tuned_pri, best_chrom, best_fit, best_outer_history


# -----------------------------
# Final comparative analysis (11+)
# -----------------------------
def fuzzy_priority_tuned(activity, visibility, act_params, vis_params):
    aA, bA, cA = act_params
    aV, bV, cV = vis_params
    act_hi = trimf(activity, aA, bA, cA)
    vis_po = trimf(visibility, aV, bV, cV)
    return np.maximum(act_hi, vis_po)


def get_schedule_from_target(target_curve, hours: int = 24):
    temp_pop = [np.random.randint(0, 3, size=hours) for _ in range(30)]
    for _ in range(50):
        fits = []
        for ind in temp_pop:
            dev = np.mean((ind / 2.0 - target_curve) ** 2)
            eng = sum(ENERGY_COST[l] for l in ind) / hours
            fits.append(0.7 * dev + 0.3 * eng)

        survivors = [temp_pop[i] for i in np.argsort(fits)[:15]]
        temp_pop = survivors.copy()
        while len(temp_pop) < 30:
            p = survivors[np.random.randint(len(survivors))].copy()
            if np.random.rand() < 0.2:
                idx = np.random.randint(hours)
                p[idx] = np.random.randint(0, 3)
            temp_pop.append(p)

    fits = []
    for ind in temp_pop:
        dev = np.mean((ind / 2.0 - target_curve) ** 2)
        eng = sum(ENERGY_COST[l] for l in ind) / hours
        fits.append(0.7 * dev + 0.3 * eng)
    return temp_pop[np.argmin(fits)]


def calculate_metrics(schedule, name, activity_hourly, visibility_hourly, hours: int = 24):
    energy_raw = sum(ENERGY_COST[l] for l in schedule)
    energy_pct = (energy_raw / hours) * 100

    risk_score = 0.0
    for h in range(hours):
        base = activity_hourly[h] * (2 - visibility_hourly[h])
        dark = DARKNESS_RISK * (2 - schedule[h])
        pen = night_penalty(h, schedule[h])
        risk_score += base + dark + pen

    return {
        "Method": name,
        "Energy (%)": energy_raw,
        "Energy_Pct": energy_pct,
        "Safety Risk": risk_score,
    }


def plot_comparative_analysis(df_results: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df_results))
    width = 0.35

    ax1.bar(x - width / 2, df_results["Energy_Pct"], width, label="Energy Usage (%)", color="skyblue")
    ax1.set_ylabel("Energy Usage (%)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, df_results["Safety Risk"], width, label="Safety Risk (Lower is Better)", color="salmon")
    ax2.set_ylabel("Safety Risk Score", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df_results["Method"])
    ax1.set_title("Trade-off Analysis: Energy vs Safety")
    ax1.grid(True, axis="y", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

    plt.tight_layout()
    plt.show()


def plot_all_schedules(baseline_schedule, best_schedule, best_fuzzy_schedule, best_tuned_schedule, hours: int = 24):
    plt.figure(figsize=(12, 4))
    plt.step(range(hours), baseline_schedule, where="mid", label="Baseline", linestyle=":", color="gray")
    plt.step(range(hours), best_schedule, where="mid", label="Standard EA", color="green", alpha=0.6)
    plt.step(range(hours), best_fuzzy_schedule, where="mid", label="Fuzzy EA", color="blue", alpha=0.6)
    plt.step(range(hours), best_tuned_schedule, where="mid", label="Tuned Fuzzy EA", color="red", linewidth=2)

    plt.yticks([0, 1, 2], ["OFF", "DIM", "FULL"])
    plt.title("Comparison of Generated Schedules")
    plt.xlabel("Hour")
    plt.ylabel("Light Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# -----------------------------
# Robust fitness (late cell)
# -----------------------------
LAMBDA_RISK = 50.0


def robust_fitness(schedule, target_curve):
    """Oblicza fitness z silną karą za ryzyko."""
    hours = len(schedule)

    energy_cost = np.sum([ENERGY_COST[l] for l in schedule])
    energy_norm = energy_cost / hours

    light_levels = schedule / 2.0
    diff = target_curve - light_levels

    risk_penalty = 0.0
    for h in range(hours):
        d = diff[h]
        if d > 0:
            risk_penalty += (d**2) * LAMBDA_RISK
        else:
            risk_penalty += (d**2) * 0.5

    risk_norm = risk_penalty / hours

    hard_penalty = 0.0
    for h, level in enumerate(schedule):
        hard_penalty += night_penalty(h, level)

    return energy_norm + risk_norm + hard_penalty


# -----------------------------
# Dynamic scheduling (13)
# -----------------------------
def solve_for_day_fast(daily_target_curve):
    return np.round(daily_target_curve * 2.0).astype(int)


def plot_dynamic_scheduling(dataset: pd.DataFrame, tuned_act, tuned_pri):
    """
    Replicates your "DYNAMIC SCHEDULING (FINAL OPTIMIZED)" plotting.
    """
    HOURS = 24

    GLOBAL_MAX_ACT = np.percentile(dataset["activity_index"], 99.5)
    if GLOBAL_MAX_ACT == 0:
        GLOBAL_MAX_ACT = 1.0

    days_of_week_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, axes = plt.subplots(7, 1, figsize=(12, 22))

    for day_idx in range(7):
        ax = axes[day_idx]
        start = day_idx * 24
        end = start + 24

        day_vis = dataset["visibility"].iloc[start:end].values
        day_act_raw = dataset["activity_index"].iloc[start:end].values

        day_vis_norm = day_vis / 2.0
        day_act_norm = day_act_raw / GLOBAL_MAX_ACT
        day_act_norm = np.clip(day_act_norm, 0.0, 1.0)

        day_target_curve = np.zeros(HOURS)
        for h in range(HOURS):
            day_target_curve[h] = fuzzy_priority_tuned(
                day_act_norm[h], day_vis_norm[h], tuned_act, tuned_pri
            )

        day_schedule = solve_for_day_fast(day_target_curve)

        day_name = days_of_week_names[int(dataset["day_of_week"].iloc[start])]
        rain = dataset["rain"].iloc[start:end].sum()
        fog = dataset["fog"].iloc[start:end].sum()

        title = f"Day {day_idx+1}: {day_name}"
        if rain > 0:
            title += f" | Rain: {rain:.1f}h"
        if fog > 0:
            title += f" | Fog: {fog:.1f}h"

        ax.fill_between(range(24), 0, day_act_norm * 2.0, color="gray", alpha=0.2, label="Activity (Norm)")

        bad_vis = day_vis < 2
        if np.any(bad_vis):
            ax.scatter(np.where(bad_vis)[0], day_vis[bad_vis], color="orange", marker="x", label="Poor Vis")

        ax.step(range(24), day_schedule, where="mid", color="red", linewidth=2.5, label="Optimized Schedule")
        ax.plot(range(24), day_target_curve * 2.0, "b--", linewidth=1, alpha=0.5, label="Fuzzy Target")

        ax.set_title(title, fontweight="bold")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["OFF", "DIM", "FULL"])
        ax.set_ylim(-0.1, 2.5)
        ax.grid(True, alpha=0.3)

        if day_idx == 0:
            ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("dynamic_scheduling.png")
    plt.show()


def run_ea_pipeline(
    csv_path: str = "street_light_dataset.csv",
    pop_size: int = 50,
    n_generations: int = 100,
    mutation_rate: float = 0.1,
    elitism_count: int = 2,
):
    """
    Convenience wrapper that runs:
      - standard EA
      - fuzzy EA
      - meta-optimization (tuned fuzzy parameters)
      - comparative analysis plots
      - dynamic scheduling plots
    """
    dataset = pd.read_csv(csv_path)

    activity_hourly, visibility_hourly, visibility_norm = build_hourly_profiles(dataset)
    baseline_schedule = build_baseline_schedule(hours=24)

    # Standard EA
    best_schedule, best_fit, best_hist = run_standard_ea(
        activity_hourly=activity_hourly,
        visibility_hourly=visibility_hourly,
        baseline_schedule=baseline_schedule,
        pop_size=pop_size,
        n_generations=n_generations,
        mutation_rate=mutation_rate,
        elitism_count=elitism_count,
    )
    plot_standard_ea_results(best_hist, baseline_schedule, best_schedule, hours=24)

    # Fuzzy EA
    plot_membership_functions()
    target_priority_curve = build_target_priority_curve(activity_hourly, visibility_norm, hours=24)

    print("Fuzzy Logic Sanity Check (Sample Hours):")
    print("-" * 60)
    for h in [0, 6, 12, 18]:
        act = activity_hourly[h]
        vis = visibility_norm[h]
        p = get_fuzzy_priority(act, vis)
        print(f"Hour {h:02d} | Act: {act:.2f} | Vis: {vis:.2f} -> Priority: {p:.2f}")

    best_fuzzy_schedule, best_fuzzy_val, best_fuzzy_hist = run_fuzzy_ea(
        target_priority_curve=target_priority_curve,
        pop_size=pop_size,
        n_generations=n_generations,
        mutation_rate=mutation_rate,
        elitism_count=elitism_count,
    )
    plot_fuzzy_results(best_fuzzy_hist, best_fuzzy_schedule, target_priority_curve, hours=24)

    # Meta-optimization (tuned fuzzy params)
    tuned_act, tuned_pri, best_chrom, best_meta_fit, best_outer_hist = run_meta_optimization(
        activity_hourly=activity_hourly,
        visibility_norm=visibility_norm,
        hours=24,
    )

    # Final comparative analysis
    tuned_target_curve = np.zeros(24)
    for h in range(24):
        tuned_target_curve[h] = fuzzy_priority_tuned(activity_hourly[h], visibility_norm[h], tuned_act, tuned_pri)

    best_tuned_schedule = get_schedule_from_target(tuned_target_curve, hours=24)

    results = []
    results.append(calculate_metrics(baseline_schedule, "Baseline", activity_hourly, visibility_hourly, hours=24))
    results.append(calculate_metrics(best_schedule, "Standard EA", activity_hourly, visibility_hourly, hours=24))
    results.append(calculate_metrics(best_fuzzy_schedule, "Fuzzy EA", activity_hourly, visibility_hourly, hours=24))
    results.append(calculate_metrics(best_tuned_schedule, "Tuned Fuzzy EA", activity_hourly, visibility_hourly, hours=24))
    df_results = pd.DataFrame(results)

    print("\nRESULTS TABLE:")
    print(df_results[["Method", "Energy_Pct", "Safety Risk"]].round(2))

    plot_comparative_analysis(df_results)
    plot_all_schedules(baseline_schedule, best_schedule, best_fuzzy_schedule, best_tuned_schedule, hours=24)

    # Dynamic scheduling
    plot_dynamic_scheduling(dataset, tuned_act, tuned_pri)

    return {
        "dataset": dataset,
        "activity_hourly": activity_hourly,
        "visibility_hourly": visibility_hourly,
        "visibility_norm": visibility_norm,
        "baseline_schedule": baseline_schedule,

        "best_schedule": best_schedule,
        "best_fuzzy_schedule": best_fuzzy_schedule,
        "best_tuned_schedule": best_tuned_schedule,
        "df_results": df_results,
        "tuned_act": tuned_act,
        "tuned_pri": tuned_pri,
    }


if __name__ == "__main__":
    run_ea_pipeline()
