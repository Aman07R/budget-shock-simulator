import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Budget Shock Simulator",
    layout="wide"
)

@st.cache_resource
def load_assets():
    model = joblib.load("budget_stress_model.pkl")
    scaler = joblib.load("budget_stress_scaler.pkl")
    return model, scaler

model, scaler = load_assets()

def risk_level(prob):
    if prob < 0.30:
        return "Low"
    elif prob < 0.70:
        return "Medium"
    return "High"

def risk_color(level):
    if level == "Low":
        return "green"
    elif level == "Medium":
        return "orange"
    return "red"

def predict_financial_stress(
    monthly_income,
    financial_aid,
    housing,
    food,
    transportation,
    books_supplies,
    entertainment,
    personal_care,
    technology,
    health_wellness,
    miscellaneous
):
    input_data = pd.DataFrame([{
        "monthly_income": monthly_income,
        "financial_aid": financial_aid,
        "housing": housing,
        "food": food,
        "transportation": transportation,
        "books_supplies": books_supplies,
        "entertainment": entertainment,
        "personal_care": personal_care,
        "technology": technology,
        "health_wellness": health_wellness,
        "miscellaneous": miscellaneous
    }])

    input_scaled = scaler.transform(input_data)
    stress_probability = float(model.predict_proba(input_scaled)[0][1])
    predicted_class = int(model.predict(input_scaled)[0])

    return predicted_class, stress_probability

def build_breakdown_df(
    housing,
    food,
    transportation,
    books_supplies,
    entertainment,
    personal_care,
    technology,
    health_wellness,
    miscellaneous
):
    return pd.DataFrame({
        "Category": [
            "Housing",
            "Food",
            "Transportation",
            "Books & Supplies",
            "Entertainment",
            "Personal Care",
            "Technology",
            "Health & Wellness",
            "Miscellaneous"
        ],
        "Amount": [
            housing,
            food,
            transportation,
            books_supplies,
            entertainment,
            personal_care,
            technology,
            health_wellness,
            miscellaneous
        ]
    })

def summary_text(total_funds, total_spending, leftover_money, stress_probability, level, top_category):
    if leftover_money < 0:
        budget_status = "Your budget is currently negative, which increases financial pressure."
    elif leftover_money < 100:
        budget_status = "Your budget is still positive, but the remaining cushion is very small."
    else:
        budget_status = "Your budget has some remaining room, which reduces immediate pressure."

    if top_category == "Housing":
        spending_driver = "Housing is the largest expense and appears to be the main driver of risk."
    else:
        spending_driver = f"{top_category} is the largest expense and is contributing most to the overall budget load."

    return (
        f"You have ${total_funds:,.0f} in total monthly funds and ${total_spending:,.0f} in total spending, "
        f"leaving ${leftover_money:,.0f} at month-end. The model estimates a {stress_probability:.1%} probability "
        f"of financial stress, placing this budget in the {level.lower()} risk category. "
        f"{budget_status} {spending_driver}"
    )

def apply_single_shock(values, shock_type, shock_amount):
    updated = values.copy()

    if shock_type == "None" or shock_amount == 0:
        return updated

    if shock_type == "Housing Increase":
        updated["housing"] += shock_amount
    elif shock_type == "Income Drop":
        updated["monthly_income"] = max(0, updated["monthly_income"] - shock_amount)
    elif shock_type == "Emergency Expense":
        updated["miscellaneous"] += shock_amount
    elif shock_type == "Food Cost Increase":
        updated["food"] += shock_amount
    elif shock_type == "Aid Reduction":
        updated["financial_aid"] = max(0, updated["financial_aid"] - shock_amount)
    elif shock_type == "Discretionary Cut":
        updated["entertainment"] = max(0, updated["entertainment"] - shock_amount)

    return updated

def short_shock_label(shock_type, shock_amount):
    if shock_type == "None" or shock_amount == 0:
        return "No Shock"

    if shock_type == "Housing Increase":
        return f"Housing +${shock_amount}"
    elif shock_type == "Income Drop":
        return f"Income -${shock_amount}"
    elif shock_type == "Emergency Expense":
        return f"Emergency +${shock_amount}"
    elif shock_type == "Food Cost Increase":
        return f"Food +${shock_amount}"
    elif shock_type == "Aid Reduction":
        return f"Aid -${shock_amount}"
    elif shock_type == "Discretionary Cut":
        return f"Entertainment -${shock_amount}"

    return shock_type


def clean_scenario_name(label):
    return label.replace("Combined: ", "Combined | ")

def evaluate_scenario(label, values):
    _, prob = predict_financial_stress(**values)
    total_funds = values["monthly_income"] + values["financial_aid"]
    total_spending = (
        values["housing"] + values["food"] + values["transportation"] +
        values["books_supplies"] + values["entertainment"] + values["personal_care"] +
        values["technology"] + values["health_wellness"] + values["miscellaneous"]
    )
    leftover_money = total_funds - total_spending

    return {
        "Scenario": label,
        "Total Funds": total_funds,
        "Total Spending": total_spending,
        "Leftover Money": leftover_money,
        "Stress Probability": prob,
        "Risk Level": risk_level(prob)
    }

def what_if_analysis(base_inputs, shock_type_1, shock_amount_1, shock_type_2, shock_amount_2):
    results = []

    results.append({
        **evaluate_scenario("Current Budget", base_inputs.copy()),
        "Chart Label": "Baseline"
    })

    has_first = shock_type_1 != "None" and shock_amount_1 > 0
    has_second = shock_type_2 != "None" and shock_amount_2 > 0

    if has_first:
        first_values = apply_single_shock(base_inputs, shock_type_1, shock_amount_1)
        first_label = format_shock_label(shock_type_1, shock_amount_1)
        first_chart_label = short_shock_label(shock_type_1, shock_amount_1)

        results.append({
            **evaluate_scenario(first_label, first_values),
            "Chart Label": first_chart_label
        })

    if has_second:
        second_values = apply_single_shock(base_inputs, shock_type_2, shock_amount_2)
        second_label = format_shock_label(shock_type_2, shock_amount_2)
        second_chart_label = short_shock_label(shock_type_2, shock_amount_2)

        results.append({
            **evaluate_scenario(second_label, second_values),
            "Chart Label": second_chart_label
        })

    if has_first and has_second:
        combined_values = base_inputs.copy()
        combined_values = apply_single_shock(combined_values, shock_type_1, shock_amount_1)
        combined_values = apply_single_shock(combined_values, shock_type_2, shock_amount_2)

        combined_label = (
            f"Combined: {format_shock_label(shock_type_1, shock_amount_1)} + "
            f"{format_shock_label(shock_type_2, shock_amount_2)}"
        )
        combined_chart_label = (
            f"Combined | {short_shock_label(shock_type_1, shock_amount_1)} + "
            f"{short_shock_label(shock_type_2, shock_amount_2)}"
        )

        results.append({
            **evaluate_scenario(combined_label, combined_values),
            "Chart Label": combined_chart_label
        })

    return pd.DataFrame(results)

st.title("Budget Shock Simulator")
st.caption("Estimate the likelihood of monthly financial stress based on your income, aid, and spending patterns.")

with st.sidebar:
    st.header("About This App")
    st.write(
        "This simulator estimates monthly financial stress risk using a trained machine learning model. "
        "Adjust the inputs and test how different budget shocks affect risk."
    )

    profile = st.selectbox(
        "Quick Start Profile",
        [
            "Custom",
            "Typical Student Budget",
            "Tight Budget",
            "High Rent Budget"
        ]
    )

    if st.button("Load Profile"):
        if profile == "Typical Student Budget":
            st.session_state.monthly_income = 1500
            st.session_state.financial_aid = 250
            st.session_state.housing = 900
            st.session_state.food = 250
            st.session_state.transportation = 120
            st.session_state.books_supplies = 60
            st.session_state.entertainment = 100
            st.session_state.personal_care = 50
            st.session_state.technology = 70
            st.session_state.health_wellness = 60
            st.session_state.miscellaneous = 90

        elif profile == "Tight Budget":
            st.session_state.monthly_income = 1200
            st.session_state.financial_aid = 200
            st.session_state.housing = 950
            st.session_state.food = 260
            st.session_state.transportation = 120
            st.session_state.books_supplies = 70
            st.session_state.entertainment = 120
            st.session_state.personal_care = 60
            st.session_state.technology = 80
            st.session_state.health_wellness = 60
            st.session_state.miscellaneous = 100

        elif profile == "High Rent Budget":
            st.session_state.monthly_income = 1600
            st.session_state.financial_aid = 250
            st.session_state.housing = 1150
            st.session_state.food = 240
            st.session_state.transportation = 100
            st.session_state.books_supplies = 60
            st.session_state.entertainment = 90
            st.session_state.personal_care = 50
            st.session_state.technology = 70
            st.session_state.health_wellness = 50
            st.session_state.miscellaneous = 80

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Monthly Budget Inputs")

    monthly_income = st.number_input("Monthly Income", min_value=0, value=st.session_state.get("monthly_income", 1500), step=50, key="monthly_income")
    financial_aid = st.number_input("Financial Aid", min_value=0, value=st.session_state.get("financial_aid", 250), step=50, key="financial_aid")
    housing = st.number_input("Housing", min_value=0, value=st.session_state.get("housing", 1000), step=50, key="housing")
    food = st.number_input("Food", min_value=0, value=st.session_state.get("food", 250), step=10, key="food")
    transportation = st.number_input("Transportation", min_value=0, value=st.session_state.get("transportation", 120), step=10, key="transportation")
    books_supplies = st.number_input("Books & Supplies", min_value=0, value=st.session_state.get("books_supplies", 60), step=10, key="books_supplies")
    entertainment = st.number_input("Entertainment", min_value=0, value=st.session_state.get("entertainment", 100), step=10, key="entertainment")
    personal_care = st.number_input("Personal Care", min_value=0, value=st.session_state.get("personal_care", 50), step=10, key="personal_care")
    technology = st.number_input("Technology", min_value=0, value=st.session_state.get("technology", 70), step=10, key="technology")
    health_wellness = st.number_input("Health & Wellness", min_value=0, value=st.session_state.get("health_wellness", 60), step=10, key="health_wellness")
    miscellaneous = st.number_input("Miscellaneous", min_value=0, value=st.session_state.get("miscellaneous", 90), step=10, key="miscellaneous")

with col2:
    st.subheader("Scenario Controls")

    shock_options = [
        "None",
        "Housing Increase",
        "Income Drop",
        "Emergency Expense",
        "Food Cost Increase",
        "Aid Reduction",
        "Discretionary Cut"
    ]

    st.markdown("**Shock Scenario 1**")
    shock_type_1 = st.selectbox(
        "Select First Shock",
        shock_options,
        index=1
    )
    shock_amount_1 = st.number_input(
        "First Shock Amount ($)",
        min_value=0,
        value=50,
        step=10
    )

    st.markdown("**Shock Scenario 2**")
    shock_type_2 = st.selectbox(
        "Select Second Shock",
        shock_options,
        index=0
    )
    shock_amount_2 = st.number_input(
        "Second Shock Amount ($)",
        min_value=0,
        value=0,
        step=10
    )

    calculate = st.button("Calculate Risk", type="primary", use_container_width=True)

if calculate:
    base_inputs = {
        "monthly_income": monthly_income,
        "financial_aid": financial_aid,
        "housing": housing,
        "food": food,
        "transportation": transportation,
        "books_supplies": books_supplies,
        "entertainment": entertainment,
        "personal_care": personal_care,
        "technology": technology,
        "health_wellness": health_wellness,
        "miscellaneous": miscellaneous
    }

    predicted_class, stress_probability = predict_financial_stress(**base_inputs)

    total_funds = monthly_income + financial_aid
    total_spending = (
        housing + food + transportation + books_supplies +
        entertainment + personal_care + technology +
        health_wellness + miscellaneous
    )
    leftover_money = total_funds - total_spending
    level = risk_level(stress_probability)

    breakdown_df = build_breakdown_df(
        housing, food, transportation, books_supplies,
        entertainment, personal_care, technology,
        health_wellness, miscellaneous
    ).sort_values("Amount", ascending=False)

    top_category = breakdown_df.iloc[0]["Category"]

    st.markdown("---")
    st.subheader("Results")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Funds", f"${total_funds:,.0f}")
    m2.metric("Total Spending", f"${total_spending:,.0f}")
    m3.metric("Leftover Money", f"${leftover_money:,.0f}")
    m4.metric("Stress Probability", f"{stress_probability:.1%}")
    m5.metric("Risk Level", level)

    st.markdown(
        f"""
        <div style="padding:14px;border-radius:12px;border:1px solid #333;margin-top:8px;margin-bottom:18px;">
            <b>Predicted Financial Stress Class:</b> {predicted_class} <br>
            <b>Risk Category:</b> <span style="color:{risk_color(level)};">{level}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    chart_col1, chart_col2 = st.columns([1.2, 1])

    with chart_col1:
        st.subheader("Budget Breakdown")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(breakdown_df["Category"], breakdown_df["Amount"])
        ax.set_title("Spending by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Amount ($)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    with chart_col2:
        st.subheader("Spending Table")
        display_df = breakdown_df.copy()
        display_df["Amount"] = display_df["Amount"].map(lambda x: f"${x:,.0f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Summary")
    st.write(
        summary_text(
            total_funds,
            total_spending,
            leftover_money,
            stress_probability,
            level,
            top_category
        )
    )

    st.subheader("What If Shock Analysis")
    shock_df = what_if_analysis(
        base_inputs,
        shock_type_1,
        shock_amount_1,
        shock_type_2,
        shock_amount_2
    )

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.markdown("**Stress Probability by Scenario**")
    prob_df = shock_df.copy().sort_values("Stress Probability", ascending=True)

    fig_prob, ax_prob = plt.subplots(figsize=(8, 4.6))
    ax_prob.barh(prob_df["Chart Label"], prob_df["Stress Probability"])

    for i, value in enumerate(prob_df["Stress Probability"]):
        ax_prob.text(min(value + 0.015, 1.01), i, f"{value:.1%}", va="center", fontsize=9)

    ax_prob.set_xlim(0, 1.05)
    ax_prob.set_xlabel("Probability")
    ax_prob.set_ylabel("")
    ax_prob.set_title("Financial Stress Risk", fontsize=11)
    ax_prob.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig_prob)

with viz_col2:
    st.markdown("**Leftover Money by Scenario**")
    money_df = shock_df.copy().sort_values("Leftover Money", ascending=True)

    fig_money, ax_money = plt.subplots(figsize=(8, 4.6))
    ax_money.barh(money_df["Chart Label"], money_df["Leftover Money"])

    for i, value in enumerate(money_df["Leftover Money"]):
        if value >= 0:
            x_position = value + 5
            ha = "left"
        else:
            x_position = value - 8
            ha = "right"
        ax_money.text(x_position, i, f"${value:,.0f}", va="center", ha=ha, fontsize=9)

    ax_money.axvline(0, linestyle="--", linewidth=1)
    ax_money.set_xlabel("Dollars")
    ax_money.set_ylabel("")
    ax_money.set_title("Leftover Money", fontsize=11)
    ax_money.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig_money)

    shock_display = shock_df.copy()
    shock_display = shock_display.drop(columns=["Chart Label"])
    shock_display["Total Funds"] = shock_display["Total Funds"].map(lambda x: f"${x:,.0f}")
    shock_display["Total Spending"] = shock_display["Total Spending"].map(lambda x: f"${x:,.0f}")
    shock_display["Leftover Money"] = shock_display["Leftover Money"].map(lambda x: f"${x:,.0f}")
    shock_display["Stress Probability"] = shock_display["Stress Probability"].map(lambda x: f"{x:.1%}")

    st.dataframe(shock_display, use_container_width=True, hide_index=True)


    worst_row = shock_df.loc[shock_df["Stress Probability"].idxmax()]
    st.subheader("Key Insight")

    scenario_name = clean_scenario_name(worst_row["Scenario"])

    st.write(
        f"The highest-risk scenario is {scenario_name}. "
        f"It results in a financial stress probability of {worst_row['Stress Probability']:.1%} "
        f"and leftover money of ${worst_row['Leftover Money']:,.0f}."
)

else:
    st.info("Enter your monthly budget inputs and click Calculate Risk to generate results.")
