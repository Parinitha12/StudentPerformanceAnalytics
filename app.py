import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient



st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>
/* ===== Overall page ===== */
.stApp {
    background: #f8f9fd;
    color: #27324a;
}

[data-testid="stHeader"] {
    background: rgba(248, 249, 253, 0.9);
}

.stMainBlockContainer {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* ===== Clean white sidebar ===== */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e9eaf2;
}

[data-testid="stSidebar"] * {
    color: #344054;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #5146a5;
}

[data-testid="stSidebar"] a {
    color: #6258c9 !important;
    text-decoration: none !important;
    font-weight: 500;
}

[data-testid="stSidebar"] a:hover {
    color: #8b5cf6 !important;
}

/* ===== Headings ===== */
h1 {
    color: #292e48;
    font-size: 2.35rem;
    font-weight: 750;
}

h2 {
    color: #5146a5;
    font-weight: 700;
}

h3 {
    color: #414766;
    font-weight: 650;
}

/* ===== Metric cards ===== */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e9eaf5;
    border-top: 4px solid #a5a0f5;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(49, 46, 129, 0.05);
    transition: transform 0.2s ease,
                box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(49, 46, 129, 0.10);
}

[data-testid="stMetricLabel"] {
    color: #69738a;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    color: #5146a5;
    font-weight: 750;
}

/* ===== Buttons ===== */
.stButton > button,
.stDownloadButton > button {
    background: #7167d8;
    color: #ffffff;
    border: 1px solid #7167d8;
    border-radius: 10px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #5b50bc;
    color: #ffffff;
    border-color: #5b50bc;
}

/* ===== Input fields ===== */
.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"] {
    background: #ffffff;
    border-radius: 9px;
}

/* ===== Tables ===== */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #ffffff;
    border: 1px solid #e7e9f2;
    border-radius: 12px;
}

/* ===== Dividers ===== */
hr {
    border-color: #e5e7f0;
    margin: 1.5rem 0;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 9px 16px;
}

.stTabs [aria-selected="true"] {
    color: #5146a5;
}

/* ===== Expanders ===== */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e7e9f2;
    border-radius: 12px;
}

/* ===== Links ===== */
a {
    color: #6258c9;
}

/* ===== Footer spacing ===== */
[data-testid="stVerticalBlock"] {
    gap: 1rem;
}
</style>
""", unsafe_allow_html=True)



st.title("📊 Student Performance Analytics")
st.markdown(
    "Monitor student grades, analyze performance trends, "
    "and explore data-driven insights."
)

st.markdown("---")

# Sidebar navigation
st.sidebar.title("📊 Student Analytics")
st.sidebar.caption("Student Performance Management")

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")

st.sidebar.markdown(
    "[🏠 Dashboard Overview](#student-performance-analytics)"
)

st.sidebar.markdown(
    "[🗄️ Database Insights](#mongodb-database-insights)"
)

st.sidebar.markdown(
    "[📈 Performance Analytics](#deeper-performance-analytics)"
)

st.sidebar.markdown(
    "[🤖 Machine Learning](#machine-learning-final-grade-estimation)"
)

st.sidebar.markdown(
    "[🔎 Student Records](#explore-student-records)"
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Python, Streamlit & MongoDB")

st.write("Analyze student grades and academic performance using MongoDB.")

@st.cache_data(ttl=10)
def load_students():
    client = MongoClient(
        "mongodb://localhost:27017/",
        serverSelectionTimeoutMS=5000
    )
    collection = client["StudentAnalyticsDB"]["students"]
    records = list(collection.find({}, {"_id": 0}))
    client.close()
    return pd.DataFrame(records)

def get_database_insights():
    client = MongoClient(
        "mongodb://localhost:27017/",
        serverSelectionTimeoutMS=5000
    )

    try:
        collection = client["StudentAnalyticsDB"]["students"]

        # Create indexes for commonly filtered fields
        collection.create_index([("school", 1)])
        collection.create_index([("G3", 1)])
        collection.create_index([("studytime", 1)])

        # Aggregation 1: Overall statistics
        overall = list(collection.aggregate([
            {
                "$group": {
                    "_id": None,
                    "total_students": {"$sum": 1},
                    "average_grade": {"$avg": "$G3"},
                    "highest_grade": {"$max": "$G3"},
                    "lowest_grade": {"$min": "$G3"},
                    "average_absences": {"$avg": "$absences"}
                }
            }
        ]))

        # Aggregation 2: Performance by school
        by_school = list(collection.aggregate([
            {
                "$group": {
                    "_id": "$school",
                    "student_count": {"$sum": 1},
                    "average_final_grade": {"$avg": "$G3"}
                }
            },
            {"$sort": {"average_final_grade": -1}}
        ]))

        # Aggregation 3: Performance by study time
        by_studytime = list(collection.aggregate([
            {
                "$group": {
                    "_id": "$studytime",
                    "student_count": {"$sum": 1},
                    "average_final_grade": {"$avg": "$G3"}
                }
            },
            {"$sort": {"_id": 1}}
        ]))

        # Aggregation 4: Count students in grade categories
        grade_categories = list(collection.aggregate([
            {
                "$project": {
                    "performance": {
                        "$switch": {
                            "branches": [
                                {
                                    "case": {"$lt": ["$G3", 10]},
                                    "then": "Below 10"
                                },
                                {
                                    "case": {"$lt": ["$G3", 15]},
                                    "then": "10 to 14"
                                }
                            ],
                            "default": "15 and above"
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": "$performance",
                    "student_count": {"$sum": 1}
                }
            }
        ]))

        return overall, by_school, by_studytime, grade_categories

    finally:
        client.close()


try:
    df = load_students()

    if df.empty:
        st.warning("No student records found in MongoDB.")
        st.stop()

    # Summary cards
    total = len(df)
    average = df["G3"].mean()
    highest = df["G3"].max()
    support_count = int((df["G3"] < 10).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", total)
    c2.metric("Average Final Grade", f"{average:.2f}/20")
    c3.metric("Highest Grade", f"{highest}/20")
    c4.metric("Below 10/20", support_count)

    st.divider()

    # Grade comparison
    st.subheader("Grade Analysis")
    grade_data = pd.DataFrame({
        "Assessment": ["First Period", "Second Period", "Final"],
        "Average Grade": [
            df["G1"].mean(),
            df["G2"].mean(),
            df["G3"].mean()
        ]
    })

    fig = px.bar(
        grade_data,
        x="Assessment",
        y="Average Grade",
        text_auto=".2f",
        title="Average Grades Across Assessments"
    )
    fig.update_yaxes(range=[0, 20])
    fig.update_layout(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(
        family="Arial, sans-serif",
        color="#414766",
        size=13
    ),
    title_font=dict(
        color="#5146a5",
        size=19
    ),
    margin=dict(l=20, r=20, t=60, b=20)
)
    st.plotly_chart(fig, use_container_width=True)

    # Study time analysis
    st.subheader("Study Time vs Final Grade")
    study_data = (
        df.groupby("studytime", as_index=False)
        .agg(
            Average_Grade=("G3", "mean"),
            Student_Count=("G3", "size")
        )
    )
    study_data["Average_Grade"] = study_data["Average_Grade"].round(2)

    fig2 = px.bar(
        study_data,
        x="studytime",
        y="Average_Grade",
        text="Average_Grade",
        hover_data=["Student_Count"],
        title="Average Final Grade by Study-Time Category",
        labels={
            "studytime": "Study-Time Category",
            "Average_Grade": "Average Final Grade"
        }
    )
    fig2.update_yaxes(range=[0, 20])
    st.plotly_chart(fig2, use_container_width=True)

    # Search and filter
   
    # MongoDB database insights
    st.divider()
    st.header("🗄️ MongoDB Database Insights")

    try:
        overall, by_school, by_studytime, grade_categories = (
            get_database_insights()
        )

        if overall:
            stats = overall[0]

            a, b, c = st.columns(3)
            a.metric(
                "Average Absences",
                f"{stats['average_absences']:.2f}"
            )
            b.metric(
                "Lowest Final Grade",
                f"{stats['lowest_grade']}"
            )
            c.metric(
                "Highest Final Grade",
                f"{stats['highest_grade']}"
            )

        st.subheader("Performance by School")
        if by_school:
            school_df = pd.DataFrame(by_school).rename(
                columns={
                    "_id": "School",
                    "student_count": "Students",
                    "average_final_grade": "Average Final Grade"
                }
            )
            school_df["Average Final Grade"] = (
                school_df["Average Final Grade"].round(2)
            )
            st.dataframe(school_df, use_container_width=True)

        st.subheader("Performance by Study Time")
        if by_studytime:
            study_df = pd.DataFrame(by_studytime).rename(
                columns={
                    "_id": "Study-Time Category",
                    "student_count": "Students",
                    "average_final_grade": "Average Final Grade"
                }
            )
            study_df["Average Final Grade"] = (
                study_df["Average Final Grade"].round(2)
            )
            st.dataframe(study_df, use_container_width=True)

        st.subheader("Final Grade Categories")
        if grade_categories:
            category_df = pd.DataFrame(grade_categories).rename(
                columns={
                    "_id": "Grade Category",
                    "student_count": "Students"
                }
            )
            st.dataframe(category_df, use_container_width=True)

        st.success(
            "MongoDB aggregations completed and indexes are available."
        )

    except Exception as e:
        st.error(f"MongoDB analysis failed: {e}")
        
    # Deeper Student Performance Analytics
    st.divider()
    st.header("📈 Deeper Performance Analytics")
    st.caption(
        "Explore relationships between study habits, absences, "
        "previous grades, and final grades."
    )

    # 1. Study time vs average final grade
    st.subheader("1. Study Time vs Final Grade")

    study_analysis = (
        df.groupby("studytime", as_index=False)
        .agg(
            Average_Final_Grade=("G3", "mean"),
            Number_of_Students=("G3", "size")
        )
    )
    study_analysis["Average_Final_Grade"] = (
        study_analysis["Average_Final_Grade"].round(2)
    )

    fig_study = px.bar(
        study_analysis,
        x="studytime",
        y="Average_Final_Grade",
        text="Average_Final_Grade",
        hover_data=["Number_of_Students"],
        labels={
            "studytime": "Study-Time Category",
            "Average_Final_Grade": "Average Final Grade"
        },
        title="Average Final Grade by Study Time"
    )
    fig_study.update_yaxes(range=[0, 20])
    st.plotly_chart(fig_study, use_container_width=True)

    st.info(
        "Study-time categories represent increasing weekly study time. "
        "Compare the averages rather than assuming study time alone "
        "determines performance."
    )

    # 2. Absences vs final grade
    st.subheader("2. Absences vs Final Grade")

    absence_data = df[["absences", "G3"]].dropna().copy()

    absence_data["Absence_Group"] = pd.cut(
        absence_data["absences"],
        bins=[-1, 0, 5, 10, 20, float("inf")],
        labels=["0", "1–5", "6–10", "11–20", "21+"]
    )

    absence_analysis = (
        absence_data.groupby(
            "Absence_Group", observed=False
        )
        .agg(
            Average_Final_Grade=("G3", "mean"),
            Number_of_Students=("G3", "size")
        )
        .reset_index()
        .dropna(subset=["Average_Final_Grade"])
    )

    absence_analysis["Average_Final_Grade"] = (
        absence_analysis["Average_Final_Grade"].round(2)
    )

    fig_absence = px.bar(
        absence_analysis,
        x="Absence_Group",
        y="Average_Final_Grade",
        text="Average_Final_Grade",
        hover_data=["Number_of_Students"],
        labels={
            "Absence_Group": "Number of Absences",
            "Average_Final_Grade": "Average Final Grade"
        },
        title="Average Final Grade by Absence Group"
    )
    fig_absence.update_yaxes(range=[0, 20])
    st.plotly_chart(fig_absence, use_container_width=True)

    # 3. Previous grades vs final grade
    st.subheader("3. Previous Grades vs Final Grade")

    previous_grade = st.selectbox(
        "Select a previous assessment",
        ["G1", "G2"],
        format_func=lambda value: {
            "G1": "First-period grade (G1)",
            "G2": "Second-period grade (G2)"
        }[value]
    )

    fig_previous = px.scatter(
        df,
        x=previous_grade,
        y="G3",
        hover_data=["school", "age", "studytime", "absences"],
        labels={
            previous_grade: "Previous Grade (out of 20)",
            "G3": "Final Grade (out of 20)"
        },
        title=f"{previous_grade} vs Final Grade (G3)"
    )
    fig_previous.update_xaxes(range=[0, 20])
    fig_previous.update_yaxes(range=[0, 20])
    st.plotly_chart(fig_previous, use_container_width=True)

    st.caption(
        "Each point represents a student. The scatter plot helps "
        "show whether previous grades tend to align with final grades."
    )
    
    # Machine Learning: Final Grade Estimation
    st.divider()
    st.header("🤖 Machine Learning — Final Grade Estimation")
    st.write(
        "A Random Forest model estimates final grades from selected "
        "student characteristics and earlier assessment grades."
    )

    feature_columns = [
        "G1", "G2", "studytime", "absences", "age",
        "Medu", "Fedu", "failures", "health"
    ]

    # Check which required columns are present
    missing_columns = [
        col for col in feature_columns + ["G3"]
        if col not in df.columns
    ]

    if missing_columns:
        st.warning(
            "Missing dataset columns: " + ", ".join(missing_columns)
        )
    else:
        # Use numeric data and remove incomplete rows
        ml_data = df[feature_columns + ["G3"]].copy()
        ml_data = ml_data.replace([np.inf, -np.inf], np.nan)
        ml_data = ml_data.dropna()

        if len(ml_data) < 20:
            st.warning(
                "Not enough complete records to train and evaluate "
                "the model reliably."
            )
        else:
            X = ml_data[feature_columns]
            y = ml_data["G3"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )

            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                min_samples_leaf=2
            )
            model.fit(X_train, y_train)

            predictions = np.clip(
                model.predict(X_test), 0, 20
            )

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            st.subheader("Model Evaluation")
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean Absolute Error", f"{mae:.2f}")
            m2.metric("Root Mean Squared Error", f"{rmse:.2f}")
            m3.metric("R² Score", f"{r2:.3f}")

            st.caption(
                "MAE and RMSE are in grade points. Lower errors are "
                "generally better. R² can be negative on test data; "
                "a higher value is generally better."
            )

            # Actual versus estimated grades
            comparison = pd.DataFrame({
                "Actual Grade": y_test.to_numpy(),
                "Estimated Grade": predictions
            })

            fig_ml = px.scatter(
                comparison,
                x="Actual Grade",
                y="Estimated Grade",
                title="Actual vs Estimated Final Grades",
                labels={
                    "Actual Grade": "Actual Final Grade",
                    "Estimated Grade": "Estimated Final Grade"
                }
            )
            fig_ml.add_shape(
                type="line",
                x0=0, y0=0, x1=20, y1=20,
                line=dict(dash="dash")
            )
            fig_ml.update_xaxes(range=[0, 20])
            fig_ml.update_yaxes(range=[0, 20])
            st.plotly_chart(fig_ml, use_container_width=True)

            # Feature importance
            importance = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig_importance = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Model Feature Importance"
            )
            st.plotly_chart(
                fig_importance, use_container_width=True
            )

            st.info(
                "Feature importance describes how the fitted model "
                "uses its inputs; it does not prove causation."
            )

            # Interactive estimate
            st.subheader("Try a Grade Estimate")
            st.caption(
                "Enter hypothetical values to explore the model. "
                "These are estimates, not guaranteed student outcomes."
            )

            with st.form("grade_estimation_form"):
                c1, c2, c3 = st.columns(3)

                with c1:
                    g1 = st.number_input(
                        "First-period grade (G1)",
                        min_value=0.0, max_value=20.0, value=10.0
                    )
                    g2 = st.number_input(
                        "Second-period grade (G2)",
                        min_value=0.0, max_value=20.0, value=10.0
                    )
                    studytime = st.selectbox(
                        "Study-time category", [1, 2, 3, 4], index=1
                    )

                with c2:
                    absences = st.number_input(
                        "Absences", min_value=0, max_value=100, value=0
                    )
                    age = st.number_input(
                        "Age", min_value=15, max_value=22, value=17
                    )
                    failures = st.selectbox(
                        "Past class failures", [0, 1, 2, 3], index=0
                    )

                with c3:
                    medu = st.selectbox(
                        "Mother's education (0–4)", [0, 1, 2, 3, 4]
                    )
                    fedu = st.selectbox(
                        "Father's education (0–4)", [0, 1, 2, 3, 4]
                    )
                    health = st.selectbox(
                        "Health (1–5)", [1, 2, 3, 4, 5], index=2
                    )

                submitted = st.form_submit_button("Estimate Final Grade")

            if submitted:
                input_row = pd.DataFrame([{
                    "G1": g1,
                    "G2": g2,
                    "studytime": studytime,
                    "absences": absences,
                    "age": age,
                    "Medu": medu,
                    "Fedu": fedu,
                    "failures": failures,
                    "health": health
                }], columns=feature_columns)

                estimate = float(
                    np.clip(model.predict(input_row)[0], 0, 20)
                )

                st.success(
                    f"Estimated final grade: {estimate:.1f} / 20"
                )
                st.caption(
                    "This is an experimental estimate based on a small "
                    "historical dataset, not a reliable individual "
                    "academic prediction."
                )



    # Advanced student filters
    st.subheader("🔎 Explore Student Records")
    st.caption("Filter the dataset to explore different student groups.")

    filtered = df.copy()

    # Final grade range
    min_grade, max_grade = st.slider(
        "Final grade range",
        min_value=0,
        max_value=20,
        value=(0, 20)
    )
    filtered = filtered[
        filtered["G3"].between(min_grade, max_grade)
    ]

    # School filter
    if "school" in filtered.columns:
        school_options = sorted(
            filtered["school"].dropna().unique().tolist()
        )
        selected_schools = st.multiselect(
            "School",
            options=school_options,
            default=school_options
        )
        filtered = filtered[
            filtered["school"].isin(selected_schools)
        ]

    # Gender filter
    if "sex" in filtered.columns:
        gender_options = sorted(
            filtered["sex"].dropna().unique().tolist()
        )
        selected_genders = st.multiselect(
            "Gender",
            options=gender_options,
            default=gender_options
        )
        filtered = filtered[
            filtered["sex"].isin(selected_genders)
        ]

    # Age filter
    if "age" in filtered.columns and not filtered.empty:
        min_age = int(filtered["age"].min())
        max_age = int(filtered["age"].max())

        age_range = st.slider(
            "Age range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )
        filtered = filtered[
            filtered["age"].between(
                age_range[0], age_range[1]
            )
        ]

    # Study-time filter
    if "studytime" in filtered.columns:
        study_options = sorted(
            filtered["studytime"].dropna().unique().tolist()
        )
        selected_studytime = st.multiselect(
            "Study-time category",
            options=study_options,
            default=study_options,
            help="Higher categories represent more weekly study time."
        )
        filtered = filtered[
            filtered["studytime"].isin(selected_studytime)
        ]

    # Show results
    st.metric("Matching student records", len(filtered))
    st.dataframe(filtered, use_container_width=True)

    # Download filtered results
    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download filtered records as CSV",
        data=csv_data,
        file_name="filtered_student_analysis.csv",
        mime="text/csv"
    )

    # Export
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered records as CSV",
        data=csv_data,
        file_name="student_analysis.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Could not load the dashboard: {e}")
    st.info(
        "Check that MongoDB is running and that "
        "StudentAnalyticsDB contains the students collection."
    )
