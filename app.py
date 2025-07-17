import streamlit as st
import pandas as pd
from itertools import combinations
from ortools.sat.python import cp_model
import datetime
from io import BytesIO

# Length of each time slot in minutes. Set via UI in the Streamlit app.
DEFAULT_SLOT_MINUTES = 60

# Predefined FT/PT templates available in the sidebar. "Custom" means the
# user-provided hour lists from the text inputs will be used.
TEMPLATES = {
    "Custom": None,
    "4x12 FT / 6,6,4,4,4 PT": {
        "ft": [12, 12, 12, 12],
        "pt": [6, 6, 4, 4, 4],
    },
    "10,10,10,10,8 FT / 6,6,4,4,4 PT": {
        "ft": [10, 10, 10, 10, 8],
        "pt": [6, 6, 4, 4, 4],
    },
}

# ------------------------ Helper functions ------------------------


def load_demand(uploaded_file: bytes, slot_minutes: int) -> pd.DataFrame:
    """Load Excel file with columns Day, Slot, Demand.

    The input may contain English headers (``Day``, ``Slot``, ``Demand``) or the
    Spanish headers ``D\u00eda``, ``Horario`` and ``Suma de Agentes Requeridos Erlang``.
    ``Horario`` values such as ``"00:00"`` are converted to slot numbers starting at ``1``.
    """

    df = pd.read_excel(uploaded_file)
    slots_per_day = 24 * 60 // slot_minutes

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
                h = v.hour
                m = v.minute
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                # Excel times may come as fractional days
                total_minutes = int(round(float(v) * 24 * 60)) % (24 * 60)
                h, m = divmod(total_minutes, 60)
            else:
                s = str(v).strip()
                parts = s.split(":")
                h = int(parts[0])
                m = int(parts[1]) if len(parts) > 1 else 0
            total_minutes = h * 60 + m
            if not 0 <= total_minutes < 24 * 60:
                raise ValueError(f"Invalid time '{v}' in Horario")
            if total_minutes % slot_minutes != 0:
                raise ValueError(
                    f"Time '{v}' does not align with {slot_minutes}-minute slots"
                )
            slot = total_minutes // slot_minutes + 1
            if not 1 <= slot <= slots_per_day:
                raise ValueError(f"Invalid time '{v}' in Horario")
            return slot

        df["Slot"] = df["Horario"].apply(_to_slot)

    required_cols = {"Day", "Slot", "Demand"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Excel must contain columns {required_cols}")

    # Validate day and slot ranges
    if not df["Day"].between(1, 7).all():
        raise ValueError("Day values must be between 1 and 7")
    if not df["Slot"].between(1, slots_per_day).all():
        raise ValueError(f"Slot values must be between 1 and {slots_per_day}")

    slot_counts = df.groupby("Day")["Slot"].nunique()
    if (slot_counts != slots_per_day).any():
        raise ValueError(f"Each day must contain {slots_per_day} unique slots")

    df = df.sort_values(["Day", "Slot"]).reset_index(drop=True)
    return df


def generate_patterns(
    df: pd.DataFrame,
    ft_daily_hours,
    pt_daily_hours,
    break_length: int,
    break_window_start: int,
    break_window_end: int,
):
    """Generate possible weekly patterns for FT and PT employees.

    ``ft_daily_hours`` and ``pt_daily_hours`` may be either a single integer or
    a list of hours for each day of the week. When a list is provided the length
    must match the number of days in ``df`` and allows each day of the pattern
    to have a different duration.
    """
    days = sorted(df["Day"].unique())
    slots = sorted(df["Slot"].unique())
    S = len(slots)
    patterns = []
    p_id = 0

    # Allow passing a single value for all days
    if isinstance(ft_daily_hours, (int, float)):
        ft_daily_hours = [int(ft_daily_hours)] * len(days)
    if isinstance(pt_daily_hours, (int, float)):
        pt_daily_hours = [int(pt_daily_hours)] * len(days)

    if len(ft_daily_hours) != len(days) or len(pt_daily_hours) != len(days):
        raise ValueError("Daily hour lists must match number of days")

    max_ft_hours = max(ft_daily_hours) if ft_daily_hours else 0
    max_pt_hours = max(pt_daily_hours) if pt_daily_hours else 0

    # Full-time patterns with break positions
    for start in range(1, S - max_ft_hours + 2):
        for brk in range(
            start + break_window_start,
            start + break_window_end - break_length + 1,
        ):
            coverage = []
            day_list = []
            daily_slots = {}
            valid = True
            for d, hrs in zip(days, ft_daily_hours):
                if hrs <= 0:
                    continue
                if start + hrs - 1 > S or brk + break_length - 1 > start + hrs - 1:
                    valid = False
                    break
                day_list.append(d)
                daily_slots[d] = hrs
                for s in range(start, start + hrs):
                    if not (brk <= s < brk + break_length):
                        coverage.append((d, s))
            if valid and day_list:
                patterns.append(
                    {
                        "id": p_id,
                        "type": "FT",
                        "days": tuple(day_list),
                        "start": start,
                        "break": brk,
                        "daily_slots": daily_slots,
                        "coverage": set(coverage),
                    }
                )
                p_id += 1

    # Part-time patterns (no break)
    for start in range(1, S - max_pt_hours + 2):
        coverage = []
        day_list = []
        daily_slots = {}
        valid = True
        for d, hrs in zip(days, pt_daily_hours):
            if hrs <= 0:
                continue
            if start + hrs - 1 > S:
                valid = False
                break
            day_list.append(d)
            daily_slots[d] = hrs
            for s in range(start, start + hrs):
                coverage.append((d, s))
        if valid and day_list:
            patterns.append(
                {
                    "id": p_id,
                    "type": "PT",
                    "days": tuple(day_list),
                    "start": start,
                    "break": None,
                    "daily_slots": daily_slots,
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

    if "ft_daily_hours_input" not in st.session_state:
        st.session_state.ft_daily_hours_input = "8"
    if "pt_daily_hours_input" not in st.session_state:
        st.session_state.pt_daily_hours_input = "4"
    if "ft_weekly_hours" not in st.session_state:
        st.session_state.ft_weekly_hours = 40
    if "pt_weekly_hours" not in st.session_state:
        st.session_state.pt_weekly_hours = 20

    if "template" not in st.session_state:
        st.session_state.template = "Custom"

    def on_template_change():
        name = st.session_state.template
        cfg = TEMPLATES.get(name)
        if cfg:
            st.session_state.ft_daily_hours_input = ", ".join(map(str, cfg["ft"]))
            st.session_state.pt_daily_hours_input = ", ".join(map(str, cfg["pt"]))
            st.session_state.ft_weekly_hours = sum(cfg["ft"])
            st.session_state.pt_weekly_hours = sum(cfg["pt"])

    slot_minutes = st.sidebar.selectbox("Minutes per Slot", [60, 30], index=0)
    st.sidebar.selectbox(
        "Template",
        list(TEMPLATES.keys()),
        key="template",
        on_change=on_template_change,
    )
    ft_daily_hours = st.sidebar.text_input(
        "FT Daily Hours (comma separated)",
        value=st.session_state.ft_daily_hours_input,
        key="ft_daily_hours_input",
    )
    pt_daily_hours = st.sidebar.text_input(
        "PT Daily Hours (comma separated)",
        value=st.session_state.pt_daily_hours_input,
        key="pt_daily_hours_input",
    )
    ft_weekly_hours = st.sidebar.number_input(
        "FT Weekly Hours", 30, 60, value=40, key="ft_weekly_hours"
    )
    pt_weekly_hours = st.sidebar.number_input(
        "PT Weekly Hours", 10, 30, value=20, key="pt_weekly_hours"
    )
    break_length = st.sidebar.number_input("Break Length (slots)", 1, 3, value=1)
    break_window_start = st.sidebar.number_input(
        "Break Window Start (slot offset)", 1, 6, value=3
    )
    break_window_end = st.sidebar.number_input(
        "Break Window End (slot offset)", break_window_start + break_length, 8, value=5
    )

    factor = 60 // slot_minutes

    uploaded_file = st.file_uploader("Upload Demand Excel", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df = load_demand(uploaded_file, slot_minutes)
            st.write("Demand data", df)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

        if st.button("Solve"):
            days_cnt = len(df["Day"].unique())

            def _parse_list(val):
                if isinstance(val, (int, float)):
                    return [int(val)] * days_cnt
                if isinstance(val, list):
                    nums = [int(v) for v in val]
                else:
                    parts = [p.strip() for p in str(val).split(",") if p.strip()]
                    nums = [int(p) for p in parts]
                if len(nums) < days_cnt:
                    nums.extend([0] * (days_cnt - len(nums)))
                return nums[:days_cnt]

            selected_template = st.session_state.template
            cfg = TEMPLATES.get(selected_template)
            if cfg is None:
                ft_list = _parse_list(ft_daily_hours)
                pt_list = _parse_list(pt_daily_hours)
            else:
                ft_list = _parse_list(cfg["ft"])
                pt_list = _parse_list(cfg["pt"])

            ft_slots = [h * factor for h in ft_list]
            pt_slots = [h * factor for h in pt_list]

            patterns = generate_patterns(
                df,
                ft_slots,
                pt_slots,
                break_length,
                break_window_start,
                break_window_end,
            )

            used, obj = solve_schedule(df, patterns)

            if used is None:
                st.error("No feasible solution found for the selected template")
                return

            st.success(
                f"Template '{selected_template}' employees required: {int(obj)}"
            )
            st.info(
                f"Generated {len(patterns)} patterns for template '{selected_template}'"
            )
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

            # --- Detailed schedule with employee IDs ---
            schedule_rows = []
            emp_id = 1
            for u in used:
                pattern = u["pattern"]
                for _ in range(u["count"]):
                    eid = f"E{emp_id:03d}"
                    for d in pattern["days"]:
                        daily_slots = pattern["daily_slots"][d]
                        end = pattern["start"] + daily_slots - 1
                        schedule_rows.append(
                            {
                                "EmployeeID": eid,
                                "Day": d,
                                "Start": pattern["start"],
                                "End": end,
                                "BreakStart": (
                                    pattern["break"] if pattern["type"] == "FT" else None
                                ),
                            }
                        )
                    emp_id += 1
            schedule_df = pd.DataFrame(schedule_rows)
            st.dataframe(schedule_df)

            csv = schedule_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download schedule CSV",
                csv,
                "schedule.csv",
                "text/csv",
            )

            xlsx = BytesIO()
            schedule_df.to_excel(xlsx, index=False)
            st.download_button(
                "Download schedule Excel",
                xlsx.getvalue(),
                "schedule.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # --- Compute scheduled coverage per (Day, Slot) ---
            cov_df = df.copy()
            cov_df["Scheduled"] = 0
            for u in used:
                pattern = u["pattern"]
                count = u["count"]
                for d, s in pattern["coverage"]:
                    cov_df.loc[(cov_df["Day"] == d) & (cov_df["Slot"] == s), "Scheduled"] += count

            cov_df["Coverage %"] = cov_df["Scheduled"] / cov_df["Demand"] * 100

            st.write("Coverage by Day and Slot", cov_df)

            # --- Visualization of coverage percentage ---
            pivot = cov_df.pivot(index="Slot", columns="Day", values="Coverage %")
            st.line_chart(pivot)

            # --- Efficiency summary ---
            demand_hours = cov_df["Demand"].sum() * slot_minutes / 60
            demand_met = (
                cov_df[["Scheduled", "Demand"]]
                .apply(lambda r: min(r["Scheduled"], r["Demand"]), axis=1)
                .sum() * slot_minutes / 60
            )
            agent_hours = cov_df["Scheduled"].sum() * slot_minutes / 60

            eff_coverage = demand_met / demand_hours * 100 if demand_hours else 0
            eff_util = demand_met / agent_hours * 100 if agent_hours else 0

            st.metric(
                "Demand Hours Covered",
                f"{demand_met:.1f} / {demand_hours}",
                f"{eff_coverage:.1f}%",
            )
            st.metric(
                "Agent Hour Utilization",
                f"{demand_met:.1f} / {agent_hours}",
                f"{eff_util:.1f}%",
            )


if __name__ == "__main__":
    main()
