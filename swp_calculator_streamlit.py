# swp_calculator_streamlit.py
# Run: pip install streamlit pandas plotly requests numpy
# Then: streamlit run swp_calculator_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.express as px
from io import StringIO

# ------------------------
# Helper: format rupee amounts into Indian numbering system
# ------------------------
def format_inr(amount):
    """Return human-readable INR values like ₹10K, ₹5.2L, ₹3.1Cr."""
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return str(amount)

    abs_amt = abs(amt)
    if abs_amt >= 1_00_00_000:
        val = amt / 1_00_00_000
        suffix = "Cr"
    elif abs_amt >= 1_00_000:
        val = amt / 1_00_000
        suffix = "L"
    elif abs_amt >= 1_000:
        val = amt / 1_000
        suffix = "K"
    else:
        val = amt
        suffix = ""
    return f"₹{val:,.2f}{suffix}"


st.set_page_config(page_title="Mutual Fund SWP Calculator", layout="wide")



API_BASE = "https://api.mfapi.in"

@st.cache_data(ttl=86400)
def fetch_and_cache_all_mfs():
    """Fetch list of all MF schemes, save to CSV grouped by AMC, return dataframe."""
    url = f"{API_BASE}/mf"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)  # schemeName, schemeCode

    # --- Extract AMC (fund house) from schemeName ---
    # Many scheme names follow "AMC Scheme-Name ..." format
    df["fundHouse"] = df["schemeName"].str.split(" ").str[0]

    # Create display label for dropdown
    df["label"] = df["fundHouse"] + " → " + df["schemeName"]

    # Save locally
    df.to_csv("mf_list.csv", index=False)

    return df

def load_mf_list():
    """Load MF list from CSV if available, otherwise fetch new."""
    try:
        return pd.read_csv("mf_list.csv")
    except FileNotFoundError:
        # CSV not present — try fetching from API, but handle network failures gracefully
        try:
            df = fetch_and_cache_all_mfs()
            return df
        except Exception as e:
            # Inform user in the Streamlit UI and return an empty dataframe with expected columns
            st.error(
                "Could not download mutual fund list from the API."
                f" Error: {e}.\n\nYou can place a pre-downloaded `mf_list.csv` next to this script as a fallback."
            )
            cols = ["schemeName", "schemeCode", "fundHouse", "label"]
            st.info("If you have an offline `mf_list.csv`, place it next to this script and reload the app.")
            return pd.DataFrame(columns=cols)


@st.cache_data(ttl=60*60*24)
def fetch_scheme_nav_history(scheme_code):
    """Fetch full NAV history + metadata for a given scheme."""
    url = f"{API_BASE}/mf/{scheme_code}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        st.error(f"Failed to fetch NAVs for scheme {scheme_code}: {e}")
        return {}, pd.DataFrame(columns=["date", "nav"])

    # Extract metadata safely
    meta = raw.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    data = raw.get("data", [])
    rows = []
    for r in data:
        # Handle possible variations in keys
        d = r.get("date") or r.get("nav_date") or r.get("Date")
        nav_str = r.get("nav") or r.get("close") or r.get("NAV")
        if not d or not nav_str:
            continue

        try:
            nav = float(str(nav_str).replace(",", "").strip())
            try:
                try:
                    dt = datetime.strptime(d, "%d-%b-%Y")
                except ValueError:
                    dt = datetime.strptime(d, "%d-%m-%Y")
            except Exception:
                continue
            rows.append({"date": dt, "nav": nav})
        except Exception:
            continue

    df_nav = pd.DataFrame(rows)
    df_nav = df_nav.sort_values("date").reset_index(drop=True)

    return meta, df_nav

# ------------------------
# Withdrawal date generator
# ------------------------
def generate_withdrawal_dates(start_date, frequency, end_date=None, max_iter=1000):
    """
    Generate a list of withdrawal dates based on the chosen frequency.
    start_date: datetime — first withdrawal date
    frequency: str — "Monthly", "Quarterly", "Yearly", "Days:7 (Weekly)", or "Days:15"
    end_date: datetime or None — if not None, generator stops before this date
    max_iter: int — maximum number of iterations to prevent infinite loops
    """
    dates = []
    current = start_date

    if frequency == "Monthly":
        delta = relativedelta(months=1)
    elif frequency == "Quarterly":
        delta = relativedelta(months=3)
    elif frequency == "Yearly":
        delta = relativedelta(years=1)
    elif frequency == "Days:7 (Weekly)":
        delta = timedelta(days=7)
    elif frequency == "Days:15":
        delta = timedelta(days=15)
    else:
        return dates  # unknown frequency

    while len(dates) < max_iter:
        dates.append(current)
        if end_date and current >= end_date:
            break
        if frequency in ["Monthly", "Quarterly", "Yearly"]:
            current = current + delta
        elif frequency == "Days:7 (Weekly)":
            current = current + timedelta(days=7)
        elif frequency == "Days:15":
            current = current + timedelta(days=15)
        else:
            break  # safety

    return dates

# ------------------------
# Nearest previous NAV finder
# ------------------------
def nearest_previous_nav(nav_df, target_date):
    """
    Find the nearest previous NAV record for a given target date.
    nav_df: DataFrame with NAV data, must contain 'date' and 'nav' columns
    target_date: datetime — the date for which we want to find the nearest previous NAV
    Returns the row (Series) with the nearest NAV, or None if not found.
    """
    if nav_df.empty:
        return None
    # Ensure date column is in datetime format
    nav_df["date"] = pd.to_datetime(nav_df["date"], errors="coerce")
    target_date = pd.to_datetime(target_date, errors="coerce")

    # Filter to only include rows before the target date
    filtered = nav_df[nav_df["date"] <= target_date]

    if filtered.empty:
        return None

    # Return the last row in the filtered DataFrame (nearest previous NAV)
    return filtered.iloc[-1]


def xirr(cashflows):
    """
    Compute XIRR (annualized IRR) using Newton's method.
    cashflows: list of tuples (date (datetime), amount (float)) where amounts are cashflow (negative investment, positive withdrawals)
    Returns decimal annual rate, or None if cannot converge.
    """
    if len(cashflows) < 2:
        return None
    # reference date
    dates = np.array([ (d - cashflows[0][0]).days / 365.0 for d, _ in cashflows ])
    amounts = np.array([ a for _, a in cashflows ], dtype=float)
    def npv(rate):
        return np.sum(amounts / ((1 + rate) ** dates))
    def npv_derivative(rate):
        return np.sum(-amounts * dates / ((1 + rate) ** (dates + 1)))
    # initial guess
    guess = 0.05
    for _ in range(100):
        f = npv(guess)
        df = npv_derivative(guess)
        if df == 0:
            break
        new = guess - f/df
        if abs(new - guess) < 1e-6:
            return new
        guess = new
    return None

# --- UI ---
st.title("SWP Calculator — Indian Mutual Funds")
st.markdown("Simulate Systematic Withdrawal Plan (SWP) from a lumpsum mutual fund investment. "
            "Select a mutual fund, choose lumpsum & dates, then pick withdrawal frequency/amount.")

col1, col2 = st.columns([1.3, 1])

with col1:
    # -------------------------------
    # Load MF List (CSV or API)
    # -------------------------------
    mf_df = load_mf_list()

    # Ensure AMC grouping available
    mf_df["fundHouse"] = mf_df["fundHouse"].fillna("Unknown AMC")

    # Search & filter (moved inside col1 to remove vertical gap)
    st.header("🔍 Select Mutual Fund Scheme")

    search_text = st.text_input("Search fund (partial match allowed)", "").lower().strip()

    filtered = mf_df.copy()

    if search_text:
        search_words = search_text.split()
        mask = filtered["schemeName"].str.lower().apply(
            lambda name: all(word in name for word in search_words)
        )
        filtered = filtered[mask]

    if filtered.empty:
        st.warning("No matching schemes. Try different search.")
        st.stop()

    selected_label = st.selectbox(
        "Select Mutual Fund",
        options=filtered["label"].tolist(),
        key="mf_select_main"
    )

    selected_row = filtered[filtered["label"] == selected_label].iloc[0]

filtered = mf_df.copy()

if search_text:
    # Split search input into words
    search_words = search_text.split()
    
    # Filter rows that contain ALL of the words in schemeName
    mask = filtered["schemeName"].str.lower().apply(
        lambda name: all(word in name for word in search_words)
    )
    filtered = filtered[mask]

if filtered.empty:
    st.warning("No matching schemes. Try different search.")
    st.stop()

# Dropdown with typing support


# Extract selected scheme details
selected_row = filtered[filtered["label"] == selected_label].iloc[0]
scheme_code = selected_row["schemeCode"]
scheme_name = selected_row["schemeName"]

#st.success(f"✅ **Selected:** Selected: {scheme_name}")
#st.caption(f"AMC: {selected_row['fundHouse']}")

    # fetch NAV and meta
with st.spinner("Loading scheme NAV history..."):
        try:
            meta, nav_df = fetch_scheme_nav_history(scheme_code)
        except Exception as e:
            st.error(f"Failed to fetch NAVs: {e}")
            st.stop()

    # show basic meta: scheme name and launch date if available
scheme_name = meta.get("scheme_name", selected_row.get("schemeName", "Unknown Scheme"))
st.markdown(f"**Scheme:** {scheme_name}")
   # --- Scheme metadata display ---
fund_house = meta.get("fund_house", "Unknown Fund House")
launch_date_raw = meta.get("launch_date")

if launch_date_raw:
    try:
        try:
            launch_date = datetime.strptime(launch_date_raw, "%d-%b-%Y").strftime("%Y-%m-%d")
        except ValueError:
            launch_date = datetime.strptime(launch_date_raw, "%d-%m-%Y").strftime("%Y-%m-%d")
    except Exception:
        launch_date = launch_date_raw
else:
    # fallback: use earliest NAV date
    launch_date = nav_df["date"].min().strftime("%Y-%m-%d") if not nav_df.empty else "Unknown"

st.info(f"🏦 **Fund House:** {fund_house}\n\n📅 **Launch Date:** {launch_date}")
latest_nav_row = nav_df.iloc[-1] if not nav_df.empty else None
if latest_nav_row is not None:
        st.write(f"Latest NAV (as on {latest_nav_row['date'].strftime('%Y-%m-%d')}): ₹{latest_nav_row['nav']:.4f}")

with col2:
    st.subheader("")
    # --- Layout: Two Proper Columns ---
col_left, col_right = st.columns(2)

# ==========================
# 📌 LEFT COLUMN
# ==========================
with col_left:
    st.subheader("Lumpsum Inputs")

    lumpsum_amount = st.number_input(
        "Lumpsum amount (₹)",
        min_value=1.0,
        value=10000000.0,
        step=1000.0,
        format="%.2f"
    )
    st.caption(f"{format_inr(lumpsum_amount)}")

    min_date = nav_df["date"].min() if not nav_df.empty else datetime(2000, 1, 1)
    max_date = nav_df["date"].max() if not nav_df.empty else datetime.today()

    lumpsum_date_raw = st.date_input(
        "Lumpsum (investment) date",
        value=min_date.date(),
        min_value=min_date.date(),
        max_value=datetime.today().date()
    )
    lumpsum_date = datetime.combine(lumpsum_date_raw, datetime.min.time())


# ==========================
# 📌 RIGHT COLUMN
# ==========================
with col_right:
    st.subheader("SWP Inputs")

    withdrawal_start_date_raw = st.date_input(
        "SWP start date",
        value=(lumpsum_date + relativedelta(months=12)).date(),
        min_value=lumpsum_date.date(),
        max_value=datetime.today().date()
    )
    withdrawal_start_date = datetime.combine(withdrawal_start_date_raw, datetime.min.time())

    
    swp_choice = st.radio(
        "Withdrawal mode",
        ["Fixed amount", "Percent of current balance"]
    )

    if swp_choice == "Fixed amount":
        swp_amount = st.number_input(
            "Withdrawal amount (₹) per event",
            min_value=1.0,
            value=80000.0,
            step=500.0,
            format="%.2f"
        )
        st.caption(f"{format_inr(swp_amount)}")
        swp_pct = None
    else:
        swp_pct = st.number_input(
            "Withdraw percent (%) of *current balance value* each event",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1
        )
        swp_amount = None

    freq = st.selectbox(
        "Withdrawal frequency",
        ["Monthly", "Quarterly", "Yearly", "Days:7 (Weekly)", "Days:15"]
    )

    # No change to value — required later
    if freq.startswith("Days"):
        freq = freq  

    end_by = st.radio("End SWP by", ["End date", "Number of withdrawals"])

    if end_by == "End date":
        end_date_raw = st.date_input(
            "SWP end date (inclusive)",
            value=datetime.today().date(),
            min_value=withdrawal_start_date.date(),
            max_value=datetime.today().date()
        )
        end_date = datetime.combine(end_date_raw, datetime.min.time())
        max_withdrawals = None

    else:
        end_date = None
        max_withdrawals = st.number_input(
            "Number of withdrawals",
            min_value=1,
            value=5,
            step=1
        )

        
if selected_row is not None:
        scheme_code = selected_row["schemeCode"]
        scheme_name = selected_row["schemeName"]
        launch_date = selected_row.get("launchDate", "N/A")

       # st.success(f"✅ **Selected:** {scheme_name}")
        #st.info(f"🏦 **Fund House:** {selected_row['fundHouse']}\n📅 **Launch Date:** {launch_date}")
run_sim = st.button("Run SWP Simulation")

st.markdown("---")

if run_sim:
    # Validate NAV availability for lumpsum date
    if nav_df.empty:
        st.error("NAV history is empty for this scheme — cannot simulate.")
        st.stop()

    # Find NAV on lumpsum date (nearest previous)
    lumpsum_nav_rec = nearest_previous_nav(nav_df, lumpsum_date)
    if lumpsum_nav_rec is None:
        st.error("No NAV before or on the chosen lumpsum date. Choose an earlier lumpsum date.")
        st.stop()
    lumpsum_nav = lumpsum_nav_rec["nav"]

    # Compute initial units
    initial_units = lumpsum_amount / lumpsum_nav

    # Prepare simulation
    rows = []
    units_available = initial_units
    total_withdrawn = 0.0
    withdrawals_done = 0
    # initial cashflow: negative lumpsum at lumpsum_date for XIRR
    cashflows = [(lumpsum_date, -float(lumpsum_amount))]

    # Determine withdrawal generator
    withdrawal_dates = generate_withdrawal_dates(withdrawal_start_date, freq, end_date=end_date, max_iter=10000)

    for wd in withdrawal_dates:
        # stop if max_withdrawals reached
        if max_withdrawals is not None and withdrawals_done >= max_withdrawals:
            break
        # find NAV for this date (nearest previous)
        nav_rec = nearest_previous_nav(nav_df, wd)
        if nav_rec is None:
            # no NAV yet for date — skip (very early dates)
            # continue to next date
            continue
        nav = nav_rec["nav"]
        # current balance value before withdrawal
        balance_value = units_available * nav
        if balance_value <= 0:
            break
        # compute withdrawal amount
        if swp_choice == "Fixed amount":
            withdraw_amt = float(swp_amount)
        else:
            withdraw_amt = balance_value * (float(swp_pct) / 100.0)

        # if withdraw amount greater than balance, cap & mark final
        if withdraw_amt >= balance_value - 1e-6:
            units_sold = units_available
            cash_received = balance_value
            units_available = 0.0
            finished = True
        else:
            units_sold = withdraw_amt / nav
            units_available = units_available - units_sold
            cash_received = withdraw_amt
            finished = False

        total_withdrawn += cash_received
        withdrawals_done += 1
        cashflows.append((wd, float(cash_received)))

        rows.append({
            "withdrawal_date": wd.date(),
            "nav_date_used": nav_rec["date"].date(),
            "nav": round(nav, 4),
            "units_sold": round(units_sold, 6),
            "units_remaining": round(units_available, 6),
            "cash_withdrawn": round(cash_received, 2),
            "balance_value_after": round(units_available * nav, 2)
        })


        if finished:
            break

    # Create results DF
    df_out = pd.DataFrame(rows)
    # Add INR-formatted column for graph hover
    df_out["inr_balance_value_after"] = df_out["balance_value_after"].apply(format_inr)

    # --- Compute remaining value using final units & latest NAV ---
    nav_df = nav_df.sort_values("date").reset_index(drop=True)
    latest_nav = nav_df["nav"].iloc[-1] if not nav_df.empty else lumpsum_nav

    if not df_out.empty:
        final_units = df_out["units_remaining"].iloc[-1]
    else:
        final_units = units_available

    remaining_value = final_units * latest_nav

    # Summary
    st.subheader("Simulation Summary")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Initial lumpsum (₹)", format_inr(lumpsum_amount))
    colB.metric("Lumpsum NAV (on/nearest before date)", f"₹{lumpsum_nav:.4f}")
    colC.metric("Initial units", f"{initial_units:,.6f}")
    colD.metric("Total withdrawn (so far)", format_inr(total_withdrawn))
    st.metric("Current Value of Investment (After SWP)", format_inr(remaining_value))

    st.write(f"Remaining value at last withdrawal date: ₹{remaining_value:,.2f}   |   {format_inr(remaining_value)}")
    st.write(f"Number of withdrawals executed: {withdrawals_done}")

    # XIRR calculation (approx)
    rate = xirr(cashflows)
    if rate is not None:
        st.write(f"Approx. annualized return (XIRR) on cashflows: {rate*100:.2f}%")
    else:
        st.write("Could not compute XIRR for given cashflows.")

    # Show table & download
    st.subheader("Withdrawal Schedule & Run-down")
    if df_out.empty:
        st.info("No withdrawals were generated (check dates and NAV availability).")
    else:
        st.dataframe(df_out)
        csv = df_out.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="swp_schedule.csv", mime="text/csv")

    # Charts
    st.subheader("Charts")
    if not df_out.empty:
        # Compute cumulative cash withdrawn
        df_out["cum_cash_withdrawn"] = df_out["cash_withdrawn"].cumsum()
        # Add INR-formatted column for cumulative cash withdrawn hover
        df_out["inr_cum_cash_withdrawn"] = df_out["cum_cash_withdrawn"].apply(format_inr)
        import plotly.graph_objects as go
        
        fig = go.Figure()

        # Remaining balance line (INR formatted for hover text)
        fig.add_trace(go.Scatter(
            x=df_out["withdrawal_date"],
            y=df_out["balance_value_after"],
            mode="lines+markers",
            name="Remaining Balance (₹)",
            line=dict(color="blue"),
            text=df_out["inr_balance_value_after"],
            hovertemplate="Date: %{x}<br>Balance: %{text}<extra></extra>"
        ))

        # Cumulative cash withdrawn line (INR formatted for hover text)
        fig.add_trace(go.Scatter(
            x=df_out["withdrawal_date"],
            y=df_out["cum_cash_withdrawn"],
            mode="lines+markers",
            name="Cumulative Cash Withdrawn (₹)",
            line=dict(color="green"),
            text=df_out["inr_cum_cash_withdrawn"],
            hovertemplate="Date: %{x}<br>Cumulative Withdrawn: %{text}<extra></extra>"
        ))

        fig.update_layout(
            title="SWP: Remaining Balance vs Cumulative Withdrawals", 
            xaxis_title="Withdrawal Date", 
            yaxis_title="Amount (₹)", 
            legend=dict(x=0.01, y=0.99), 
            hovermode="x unified" 
            )

        st.plotly_chart(fig, use_container_width=True)


    # Also show NAV history snippet
    #st.subheader("NAV history (sample)")
    #st.write("Earliest 5 and latest 5 NAVs (for reference).")
    #st.dataframe(pd.concat([nav_df.head(5), nav_df.tail(5)]).reset_index(drop=True))

    st.success("Simulation complete.")
