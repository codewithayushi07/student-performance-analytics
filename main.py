import pandas as pd
import numpy as np


def calculate_grade(average):
    """Return a grade based on the student's average marks."""
    if average >= 90:
        return "A+"
    elif average >= 80:
        return "A"
    elif average >= 70:
        return "B"
    elif average >= 60:
        return "C"
    elif average >= 50:
        return "D"
    else:
        return "F"


def main():
    print("Student Performance Analytics System")
    print("Libraries imported successfully!")

    # 1. Load the dataset
    df = pd.read_csv("students.csv")

    # 2. Inspect the dataset
    print("\n--- Dataset Preview ---")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nDataset information:")
    df.info()

    # 3. Check and handle missing marks before calculations
    subjects = ["Maths", "Science", "English"]
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    for subject in subjects:
        df[subject] = df[subject].fillna(df[subject].mean())

    # 4. NumPy-based subject analysis
    print("\n--- Subject Analysis ---")
    for subject in subjects:
        marks = df[subject].to_numpy()
        print(f"{subject} average: {np.mean(marks):.2f}")
        print(f"{subject} highest marks: {np.max(marks)}")
        print(f"{subject} lowest marks: {np.min(marks)}")
        print(f"{subject} total marks: {np.sum(marks)}")
        print(f"{subject} standard deviation: {np.std(marks):.2f}")
        print(f"{subject} median: {np.median(marks)}")

    print("\nStudents scoring above 80 in Maths:",
          int(np.sum(df["Maths"].to_numpy() > 80)))

    if "Attendance" in df.columns:
        print("Average attendance:", f"{np.mean(df['Attendance'].to_numpy()):.2f}")

    # 5. Calculate total, average, and count of subjects passed
    df["Total"] = df[subjects].sum(axis=1)
    df["Average"] = df[subjects].mean(axis=1)
    df["Pass_Count"] = (df[subjects] >= 50).sum(axis=1)

    # Pass/fail rule: Pass if the student scores at least 50 in every subject.
    df["Result"] = np.where((df[subjects] >= 50).all(axis=1), "Pass", "Fail")
    df["Grade"] = df["Average"].apply(calculate_grade)

    # 6. Identify highest and lowest overall performers
    print("\n--- Overall Performance ---")
    print("Highest average:", f"{df['Average'].max():.2f}")
    print("Lowest average:", f"{df['Average'].min():.2f}")
    print("Average across all students:", f"{df['Average'].mean():.2f}")

    top_index = df["Average"].idxmax()
    low_index = df["Average"].idxmin()
    print("\nTop performer:")
    print(df.loc[top_index, ["Student_ID", "Name", "Total", "Average", "Grade"]])
    print("\nLowest performer:")
    print(df.loc[low_index, ["Student_ID", "Name", "Total", "Average", "Grade"]])

    # 7. Subject-wise analysis and result counts
    print("\n--- Subject-Wise Average ---")
    print(df[subjects].mean())

    print("\n--- Pass/Fail Summary ---")
    print(df["Result"].value_counts())

    print("\n--- Grade Summary ---")
    print(df["Grade"].value_counts().sort_index())

    print("\n--- Students Sorted by Average (Highest First) ---")
    print(df.sort_values("Average", ascending=False))

    # 8. Save the complete results at the end
    df.to_csv("student_results.csv", index=False)
    print("\nResults saved to student_results.csv")


if __name__ == "__main__":
    main()
