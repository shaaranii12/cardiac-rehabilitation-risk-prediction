"""
Feature interpretation dictionary for Cardiac Risk Prediction
Derived from SHAP directionality and clinical reasoning.
"""

FEATURE_INTERPRETATION = {

    "Heart Rate (%)": {
        "positive": (
            "Lower prescribed target heart rate percentage increased predicted cardiac risk, "
            "reflecting stricter exercise intensity limits commonly applied to higher-risk patients."
        ),
        "negative": (
            "Higher prescribed target heart rate percentage reduced predicted cardiac risk, "
            "indicating greater cardiovascular tolerance and functional capacity."
        )
    },

    "Age": {
        "positive": (
            "Advanced age contributed to increased predicted cardiac risk, "
            "reflecting age-related cardiovascular vulnerability."
        ),
        "negative": (
            "Younger age or preserved physiological capacity mitigated predicted cardiac risk."
        )
    },

    "ecg_sinus_rhythm": {
        "positive": (
            "Absence of normal sinus rhythm increased predicted cardiac risk, "
            "suggesting underlying cardiac conduction abnormalities."
        ),
        "negative": (
            "Presence of normal sinus rhythm reduced predicted cardiac risk, "
            "indicating stable cardiac electrical activity."
        )
    },

    "Exercise Habit - Frequency": {
        "positive": (
            "Low frequency of habitual exercise increased predicted cardiac risk "
            "due to reduced regular cardiovascular engagement."
        ),
        "negative": (
            "More frequent participation in exercise reduced predicted cardiac risk, "
            "reflecting improved cardiovascular conditioning."
        )
    },

    "Exercise Habit - Duration": {
        "positive": (
            "Shorter habitual exercise duration increased predicted cardiac risk, "
            "indicating lower baseline physical conditioning."
        ),
        "negative": (
            "Longer habitual exercise duration reduced predicted cardiac risk, "
            "reflecting better cardiovascular fitness."
        )
    },

    "dx_cabg": {
        "positive": (
            "History of coronary artery bypass graft surgery contributed to increased predicted cardiac risk, "
            "reflecting severe underlying coronary artery disease."
        ),
        "negative": (
            "Absence of prior bypass surgery reduced predicted cardiac risk."
        )
    },

    "Test Today - METS": {
        "positive": (
            "Lower achieved METS during exercise testing increased predicted cardiac risk, "
            "indicating reduced exercise tolerance."
        ),
        "negative": (
            "Higher achieved METS reduced predicted cardiac risk, "
            "reflecting stronger functional capacity."
        )
    },

    "muscle_power_left": {
        "positive": (
            "Reduced left-sided muscle strength modestly increased predicted cardiac risk, "
            "reflecting lower overall functional capacity."
        ),
        "negative": (
            "Preserved muscle strength modestly reduced predicted cardiac risk."
        )
    },

    "termination_adverse": {
        "positive": (
            "Adverse termination of exercise testing increased predicted cardiac risk, "
            "reflecting physiological limitation or intolerance to exertion."
        ),
        "negative": (
            "Absence of adverse test termination reduced predicted cardiac risk."
        )
    },

    "Test Today - peak HR": {
        "positive": (
            "Lower peak heart rate achieved during exercise testing increased predicted cardiac risk, "
            "suggesting limited cardiovascular response to exertion."
        ),
        "negative": (
            "Higher peak heart rate achieved during testing reduced predicted cardiac risk, "
            "indicating preserved cardiovascular responsiveness."
        )
    }

}
