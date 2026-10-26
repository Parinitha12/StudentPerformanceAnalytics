
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from pathlib import Path

# 1. Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
client.admin.command("ping")
db = client["StudentAnalyticsDB"]
students = db["students"]

print("Connected to MongoDB successfully!")

# 2. Read the student dataset
csv_path = Path(__file__).parent / "student-mat.csv"
df = pd.read_csv(csv_path, sep=";")

print("\nDataset loaded successfully!")
print("Number of students:", len(df))
print("Number of columns:", len(df.columns))
print("\nFirst five records:")
print(df.head())

# 3. Prepare data for MongoDB
df = df.where(pd.notna(df), None)
records = df.to_dict("records")

# 4. Store the dataset in MongoDB
students.delete_many({})
students.insert_many(records)

print("\nStudent records stored in MongoDB:", students.count_documents({}))

# 5. Analyze student performance
pipeline = [
    {
        "$group": {
            "_id": None,
            "average_G1": {"$avg": "$G1"},
            "average_G2": {"$avg": "$G2"},
            "average_G3": {"$avg": "$G3"},
            "total_students": {"$sum": 1}
        }
    }
]

result = list(students.aggregate(pipeline))[0]

print("\n--- STUDENT PERFORMANCE ANALYSIS ---")
print("Total students:", result["total_students"])
print("Average first-period grade (G1):", round(result["average_G1"], 2))
print("Average second-period grade (G2):", round(result["average_G2"], 2))
print("Average final grade (G3):", round(result["average_G3"], 2))

# 6. Find the top 10 students by final grade
print("\n--- TOP 10 STUDENTS ---")
for student in students.find(
    {}, {"_id": 0, "school": 1, "sex": 1, "age": 1, "G3": 1}
).sort([("G3", -1), ("G2", -1)]).limit(10):
    print(student)

# 7. Compare average final grades by study time
study_pipeline = [
    {
        "$group": {
            "_id": "$studytime",
            "average_grade": {"$avg": "$G3"},
            "student_count": {"$sum": 1}
        }
    },
    {"$sort": {"_id": 1}}
]

study_results = list(students.aggregate(study_pipeline))

print("\n--- STUDY TIME ANALYSIS ---")
for item in study_results:
    print(
        "Study time category:", item["_id"],
        "| Students:", item["student_count"],
        "| Average final grade:", round(item["average_grade"], 2)
    )

# 8. Identify students who need academic support
print("\n--- STUDENTS NEEDING SUPPORT (G3 < 10) ---")
low_performers = list(
    students.find(
        {"G3": {"$lt": 10}},
        {"_id": 0, "school": 1, "sex": 1, "age": 1, "studytime": 1, "G3": 1}
    ).limit(20)
)

print("Showing up to 20 students with final grades below 10:")
for student in low_performers:
    print(student)

# 9. Create a graph of average grades
grade_labels = ["G1", "G2", "G3"]
grade_averages = [
    result["average_G1"],
    result["average_G2"],
    result["average_G3"]
]

plt.figure(figsize=(8, 5))
plt.bar(grade_labels, grade_averages)
plt.title("Average Student Grades")
plt.xlabel("Assessment")
plt.ylabel("Average Grade (out of 20)")
plt.ylim(0, 20)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "average_student_grades.png")
plt.show()

print("\nAnalysis complete! Graph saved as average_student_grades.png")

# 10. Count all students needing academic support
low_count = students.count_documents({"G3": {"$lt": 10}})
print("\nTotal students scoring below 10:", low_count)

# 11. Graph: Study time vs average final grade
study_categories = [item["_id"] for item in study_results]
study_averages = [item["average_grade"] for item in study_results]

plt.figure(figsize=(8, 5))
plt.bar(study_categories, study_averages)
plt.title("Study Time vs Average Final Grade")
plt.xlabel("Study Time Category (1 = Lowest, 4 = Highest)")
plt.ylabel("Average Final Grade (out of 20)")
plt.xticks(study_categories)
plt.ylim(0, 20)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "study_time_analysis.png")
plt.show()

print("Study-time graph saved as study_time_analysis.png")


client.close()
