# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:14:02 2026

@author: Dell
"""

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(r"D:\OneDrive\Desktop\food\food_demand.csv")


# =========================================================
# LOAD TRAINED RANDOM FOREST MODEL
# =========================================================

model = joblib.load("food_demand_random_forest_classifier.pkl")


# =========================================================
# CATEGORICAL COLUMNS
# =========================================================

categorical_columns = [
    "category",
    "cuisine",
    "center_type"
]


# =========================================================
# CREATE ENCODERS
# =========================================================

encoders = {}

for col in categorical_columns:

    values = sorted(
        df[col].astype(str).unique()
    )

    encoders[col] = {
        value: index
        for index, value in enumerate(values)
    }


# =========================================================
# COMMON DATA FOR FRONTEND
# =========================================================

def get_dropdown_data():

    return {
        "categories": sorted(
            df["category"].astype(str).unique()
        ),

        "cuisines": sorted(
            df["cuisine"].astype(str).unique()
        ),

        "center_types": sorted(
            df["center_type"].astype(str).unique()
        )
    }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    dropdown_data = get_dropdown_data()

    return render_template(
        "index.html",
        categories=dropdown_data["categories"],
        cuisines=dropdown_data["cuisines"],
        center_types=dropdown_data["center_types"]
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------------------------
        # GET ALL 14 INPUTS
        # -------------------------------------------------

        data = {

            "id": int(
                request.form["id"]
            ),

            "week": int(
                request.form["week"]
            ),

            "center_id": int(
                request.form["center_id"]
            ),

            "meal_id": int(
                request.form["meal_id"]
            ),

            "checkout_price": float(
                request.form["checkout_price"]
            ),

            "base_price": float(
                request.form["base_price"]
            ),

            "emailer_for_promotion": int(
                request.form["emailer_for_promotion"]
            ),

            "homepage_featured": int(
                request.form["homepage_featured"]
            ),

            "category": request.form["category"],

            "cuisine": request.form["cuisine"],

            "city_code": int(
                request.form["city_code"]
            ),

            "region_code": int(
                request.form["region_code"]
            ),

            "center_type": request.form["center_type"],

            "op_area": float(
                request.form["op_area"]
            )
        }


        # -------------------------------------------------
        # CREATE DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame([data])


        # -------------------------------------------------
        # ENCODE CATEGORICAL VARIABLES
        # -------------------------------------------------

        for col in categorical_columns:

            value = str(
                input_data.loc[0, col]
            )

            if value not in encoders[col]:

                dropdown_data = get_dropdown_data()

                return render_template(
                    "index.html",
                    error=f"Invalid value for {col}.",

                    categories=dropdown_data["categories"],
                    cuisines=dropdown_data["cuisines"],
                    center_types=dropdown_data["center_types"]
                )

            input_data[col] = encoders[col][value]


        # -------------------------------------------------
        # FEATURE ORDER
        #
        # MUST BE THE SAME AS TRAINING
        # -------------------------------------------------

        feature_columns = [

            "id",
            "week",
            "center_id",
            "meal_id",
            "checkout_price",
            "base_price",
            "emailer_for_promotion",
            "homepage_featured",
            "category",
            "cuisine",
            "city_code",
            "region_code",
            "center_type",
            "op_area"
        ]


        input_data = input_data[
            feature_columns
        ]


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]

        prediction = int(prediction)


        # -------------------------------------------------
        # DEMAND LABEL
        # -------------------------------------------------

        demand_labels = {

            0: "Low Demand",

            1: "Medium Demand",

            2: "High Demand"
        }


        demand = demand_labels.get(
            prediction,
            "Unknown"
        )


        # -------------------------------------------------
        # PREDICTION PROBABILITY
        # -------------------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            probability = round(
                max(probabilities) * 100,
                2
            )


        # -------------------------------------------------
        # SEND RESULT TO FRONTEND
        # -------------------------------------------------

        dropdown_data = get_dropdown_data()

        return render_template(

            "index.html",

            prediction=demand,

            probability=probability,

            submitted=True,

            categories=dropdown_data["categories"],

            cuisines=dropdown_data["cuisines"],

            center_types=dropdown_data["center_types"]
        )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        dropdown_data = get_dropdown_data()

        return render_template(

            "index.html",

            error=str(e),

            categories=dropdown_data["categories"],

            cuisines=dropdown_data["cuisines"],

            center_types=dropdown_data["center_types"]
        )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )