# OR_T

This Streamlit application helps design workforce schedules with the help of Google's OR-Tools.
It automatically tests many shift templates and computes the best combination of full-time and
part-time patterns so that demand coverage is maximized with the minimum number of employees.

## Setup

1. Install **Python 3.10** or newer.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the web UI:
   ```bash
   streamlit run app.py
   ```

## Demand data

Upload the staffing demand as an Excel file. The spreadsheet must contain the
columns `Day`, `Slot` and `Demand`. The loader also accepts the Spanish
headers `Día`, `Horario` and `Suma de Agentes Requeridos Erlang`.
Slots are numbered starting at 1 and depend on the slot length chosen in the
sidebar.

Once the file is uploaded you can experiment with the predefined shift
templates or provide your own daily hours. The application generates all
possible patterns, solves the coverage model and reports the resulting
schedule and coverage percentage.

The sidebar also contains fields for the maximum number of hours a full-time
or part-time employee may work in a week.  When patterns are generated these
weekly limits are enforced so that no proposed schedule exceeds the allowed
hours.
