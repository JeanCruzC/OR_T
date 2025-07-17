import streamlit as st
import pandas as pd
from itertools import combinations
from ortools.sat.python import cp_model
import datetime

# ------------------------ Helper functions ------------------------

def load_demand(uploaded_file: bytes) -> pd.DataFrame:
    """Load Excel file with columns Day, Slot, Demand.

    The input may contain English headers (``Day``, ``Slot``, ``Demand``) or the
    Spanish headers ``D\u00eda``, ``Horario`` and ``Suma de Agentes Requeridos Erlang``.
    ``Horario`` values such as ``"00:00"`` are converted to slot numbers ``1`` through ``24``.
    """

    df = pd.read_excel(uploaded_file)

    # Map Spanish column names to English
    col_map = {
        "D\u00eda": "Day",
        "Suma de Agentes Requeridos Erlang": "Demand",
    }
    df = df.rename(columns=col_map)

    # Parse "Horario" column to slot number if present
    if "Slot" not in df.columns and "Horario" in df.columns:
        def _to_slot(v):
            if pd.isna(v):
                raise ValueError("Horario contains invalid values")
            if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.time)):
                hour = v.hour
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                # Excel times may come as fractional days
                hour = int(float(v) * 24) % 24
            else:
                s = str(v).strip()
                hour = int(s.split(":")[0])
            slot = hour + 1
            if not 1 <= slot <= 24:
                raise ValueError(f"Invalid hour '{v}' in Horario")
            return slot

        df["Slot"] = df["Horario"].apply(_to_slot)

    required_cols = {"Day", "Slot", "Demand"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Excel must contain columns {required_cols}")

    # Validate day and slot ranges
    if not df["Day"].between(1, 7).all():
        raise ValueError("Day values must be between 1 and 7")

    slot_counts = df.groupby("Day")["Slot"].nunique()
    if (slot_counts != 24).any():
        raise ValueError("Each day must contain 24 unique slots")

    df = df.sort_values(["Day", "Slot"]).reset_index(drop=True)
    return df


def generate_patterns(
    df: pd.DataFrame,
    ft_daily_hours: int,
    pt_daily_hours: int,
    ft_weekly_hours: int,
    pt_weekly_hours: int,
    break_length: int,
    break_window_start: int,
    break_window_end: int,
):
    """Generate possible weekly patterns for FT and PT employees."""
    days = sorted(df["Day"].unique())
    slots = sorted(df["Slot"].unique())
    S = len(slots)
    patterns = []
    p_id = 0

    def day_combos(num_days):
        return list(combinations(days, num_days)) if num_days <= len(days) else []

    ft_days = ft_weekly_hours // ft_daily_hours
    pt_days = pt_weekly_hours // pt_daily_hours

    # Full-time patterns with break positions
    for combo in day_combos(ft_days):
        for start in range(1, S - ft_daily_hours + 2):
            for brk in range(start + break_window_start, start + break_window_end - break_length + 2):
                coverage = []
                for d in combo:
                    for s in range(start, start + ft_daily_hours):
                        if not (brk <= s < brk + break_length):
                            coverage.append((d, s))
                patterns.append(
                    {
                        "id": p_id,
                        "type": "FT",
                        "days": combo,
                        "start": start,
                        "break": brk,
                        "coverage": set(coverage),
                    }
                )
                p_id += 1

    # Part-time patterns (no break)
    for combo in day_combos(pt_days):
        for start in range(1, S - pt_daily_hours + 2):
            coverage = []
            for d in combo:
                for s in range(start, start + pt_daily_hours):
                    coverage.append((d, s))
            patterns.append(
                {
                    "id": p_id,
                    "type": "PT",
                    "days": combo,
                    "start": start,
                    "break": None,
                    "coverage": set(coverage),
                }
            )
            p_id += 1

    return patterns


def solve_schedule(demand_df: pd.DataFrame, patterns):
    """Solve set covering model to minimize employee count."""
    model = cp_model.CpModel()
    x = {}
    for p in patterns:
        x[p["id"]] = model.NewIntVar(0, 1000, f"p_{p['id']}")

    # Demand constraints
    for _, row in demand_df.iterrows():
        d = int(row["Day"])
        s = int(row["Slot"])
        demand = int(row["Demand"])
        model.Add(
            sum(x[p["id"]] for p in patterns if (d, s) in p["coverage"]) >= demand
        )

    model.Minimize(sum(x[p["id"]] for p in patterns))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None

    used = []
    for p in patterns:
        count = solver.Value(x[p["id"]])
        if count > 0:
            used.append({"pattern": p, "count": count})
    return used, solver.ObjectiveValue()


# ------------------------ Streamlit App ------------------------

def main():
    st.title("Workforce Scheduling with OR-Tools")

    st.sidebar.header("Configuration")
    ft_daily_hours = st.sidebar.number_input("FT Daily Hours", 6, 12, value=8)
    pt_daily_hours = st.sidebar.number_input("PT Daily Hours", 2, 8, value=4)
    ft_weekly_hours = st.sidebar.number_input("FT Weekly Hours", 30, 60, value=40)
    pt_weekly_hours = st.sidebar.number_input("PT Weekly Hours", 10, 30, value=20)
    break_length = st.sidebar.number_input("Break Length (slots)", 1, 3, value=1)
    break_window_start = st.sidebar.number_input(
        "Break Window Start (slot offset)", 1, 6, value=3
    )
    break_window_end = st.sidebar.number_input(
        "Break Window End (slot offset)", break_window_start + break_length, 8, value=5
    )

    uploaded_file = st.file_uploader("Upload Demand Excel", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df = load_demand(uploaded_file)
            st.write("Demand data", df)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

        if st.button("Solve"):
            with st.spinner("Generating patterns..."):
                patterns = generate_patterns(
                    df,
                    ft_daily_hours,
                    pt_daily_hours,
                    ft_weekly_hours,
                    pt_weekly_hours,
                    break_length,
                    break_window_start,
                    break_window_end,
                )
            st.success(f"Generated {len(patterns)} patterns")

            with st.spinner("Solving model..."):
                used, obj = solve_schedule(df, patterns)

            if used is None:
                st.error("No feasible solution found")
            else:
                st.success(f"Minimum employees required: {int(obj)}")
                res_rows = []
                for u in used:
                    p = u["pattern"]
                    res_rows.append(
                        {
                            "Type": p["type"],
                            "Days": ",".join(map(str, p["days"])),
                            "Start": p["start"],
                            "Break": p["break"],
                            "Count": u["count"],
                        }
                    )
                st.dataframe(pd.DataFrame(res_rows))


if __name__ == "__main__":
    main()
