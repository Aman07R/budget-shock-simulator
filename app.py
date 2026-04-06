# -*- coding: utf-8 -*-
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
    breakdown_df = pd.DataFrame({
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
    return breakdown_df

def plain_summary(total_funds, total_spending, leftover_money, stress_probability, level, top_category, top_amount):
    if leftover_money < 0:
        situation = "Your budget is currently running negative, which suggests a higher chance of financial stress."
    elif leftover_money < 100:
        situation = "Your budget is still positive, but the cushion is very small, so even a minor shock could create pressure."
    else:
        situation = "Your budget has some remaining room, which lowers immediate financial pressure."

    if top_category == "Housing":
        driver = "Housing is your biggest expense, which is often the main source of budget pressure."
    else:
        driver = f"{top_category} is currently your biggest expense and is playing a major role in your monthly budget."

    return (
        f"Your total monthly funds are ${total_funds:,.0f}, while total spending is ${total_spending:,.0f}. "
        f"That leaves you with ${leftover_money:,.0f} at the end of the month. "
        f"The model estimates a {stress_probability:.1%} probability of financial stress, which falls into the {level.lower()} risk category. "
        f"{situation} {driver} Your largest spending category is {top_category} at ${top_amount:,.0f}."
    )

def what_if_analysis(base_inputs, shock_type):
    scenarios = []

    baseline = base_inputs.copy()
    scenarios.append(("Current Budget", baseline))

    if shock_type == "Housing Increase":
        s1 = base_inputs.copy()
        s1["housing"] += 150
        scenarios.append(("Housing +$150", s1))

        s2 = base_inputs.copy()
        s2["housing"] += 300
        scenarios.append(("Housing +$300", s2))

    elif shock_type == "Income Drop":
        s1 = base_inputs.copy()
        s1["monthly_income"] = max(0, s1["monthly_income"] - 200)
        scenarios.append(("Income -$200", s1))

        s2 = base_inputs.copy()
        s2["monthly_income"] = max(0, s2["monthly_income"] - 400)
        scenarios.append(("Income -$400", s2))

    elif shock_type == "Emergency Expense":
        s1 = base_inputs.copy()
        s1["miscellaneous"] += 150
        scenarios.append(("Emergency +$150", s1))

        s2 = base_inputs.copy()
        s2["miscellaneous"] += 300
        scenarios.append(("Emergency +$300", s2))

    elif shock_type == "Discretionary Cut":
        s1 = base_inputs.copy()
        s1["entertainment"] = max(0, s1["entertainment"] - 50)
        scenarios.append(("Entertainment -$50", s1))

        s2 = base_inputs.copy()
        s2["entertainment"] = max(0, s2["entertainment"] - 100)
        scenarios.append(("Entertainment -$100", s2))

    results = []
    for label, values in scenarios:
        _, prob = predict_financial_stress(**values)
        total_funds = values["monthly_income"] + values["financial_aid"]
        total_spending = (
            values["housing"] + values["food"] + values["transportation"] +
            values["books_supplies"] + values["entertainment"] + values["personal_care"] +
            values["technology"] + values["health_wellness"] + values["miscellaneous"]
        )
        leftover_money = total_funds - total_spending

        results.append({
            "Scenario": label,
            "Total Funds": total_funds,
            "Total Spending": total_spending,
            "Leftover Money": leftover_money,
            "Stress Probability": prob,
            "Risk Level": risk_level(prob)
        })

    return pd.DataFrame(results)

st.title("Budget Shock Simulator")
st.caption("Estimate the likelihood of monthly financial stress based on your income, aid, and spending patterns.")

with st.sidebar:
    st.header("About This App")
    st.write(
        "This simulator estimates monthly financial stress risk using a trained machine learning model. "
        "Adjust the inputs and see how budget shocks affect risk."
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
    shock_type = st.selectbox(
        "What-If Analysis",
        ["Housing Increase", "Income Drop", "Emergency Expense", "Discretionary Cut"]
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
    top_amount = breakdown_df.iloc[0]["Amount"]

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

    st.subheader("Plain-English Summary")
    st.write(
        plain_summary(
            total_funds,
            total_spending,
            leftover_money,
            stress_probability,
            level,
            top_category,
            top_amount
        )
    )

    st.subheader("What-If Shock Analysis")
    shock_df = what_if_analysis(base_inputs, shock_type)

    shock_chart_df = shock_df.copy()
    st.bar_chart(
        shock_chart_df.set_index("Scenario")["Stress Probability"],
        use_container_width=True
    )

    shock_display = shock_df.copy()
    shock_display["Total Funds"] = shock_display["Total Funds"].map(lambda x: f"${x:,.0f}")
    shock_display["Total Spending"] = shock_display["Total Spending"].map(lambda x: f"${x:,.0f}")
    shock_display["Leftover Money"] = shock_display["Leftover Money"].map(lambda x: f"${x:,.0f}")
    shock_display["Stress Probability"] = shock_display["Stress Probability"].map(lambda x: f"{x:.1%}")

    st.dataframe(shock_display, use_container_width=True, hide_index=True)

    worst_row = shock_df.loc[shock_df["Stress Probability"].idxmax()]
    st.subheader("Key Insight")
    st.markdown(
        f"In the **{shock_type.lower()}** simulation, the highest-risk case is "
        f"**{worst_row['Scenario']}**, with a predicted financial stress probability of "
        f"**{worst_row['Stress Probability']:.1%}** and leftover money of "
        f"**${worst_row['Leftover Money']:,.0f}**."
    )
    

else:
    st.info("Enter your monthly budget inputs and click Calculate Risk to generate results.")
