import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_classification_data
from src.model import GradientBoostingModel

st.set_page_config(page_title="Gradient Boosting", page_icon="🚀", layout="wide")
st.title("🚀 Gradient Boosting Classifier")
st.markdown("Sequential ensemble with learning rate control and GridSearchCV tuning.")

dataset_name = st.selectbox("Dataset", ["wine", "iris", "breast_cancer"])
X_train, X_test, y_train, y_test, fn, cn = load_classification_data(dataset_name)

tab1, tab2 = st.tabs(["Train", "Compare"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        n_trees = st.slider("Boosting Stages", 10, 500, 100, 10)
    with c2:
        lr = st.select_slider("Learning Rate", [0.001, 0.01, 0.05, 0.1, 0.2, 0.5], 0.1)
    with c3:
        max_depth = st.slider("Max Depth", 1, 10, 3)
        tune = st.checkbox("GridSearchCV")

    if st.button("Train GB", type="primary"):
        with st.spinner("Training..."):
            model = GradientBoostingModel(n_estimators=n_trees, learning_rate=lr, max_depth=max_depth)
            model.fit(X_train, y_train, fn, cn)
            results = model.evaluate(X_test, y_test)
        st.success(f"Accuracy: **{results['accuracy']:.2%}**")
        if tune:
            with st.spinner("Grid search..."):
                best = model.grid_search(X_train, y_train)
            st.info(f"Best: {best['best_params']} → {best['best_score']:.3f}")
        imp = model.get_feature_importance()
        st.bar_chart(pd.Series(imp).sort_values(ascending=True))

with tab2:
    st.subheader("Learning Rate vs Accuracy")
    if st.button("Compare Learning Rates", type="primary"):
        results = []
        for rate in [0.01, 0.05, 0.1, 0.2, 0.5]:
            m = GradientBoostingModel(n_estimators=100, learning_rate=rate)
            m.fit(X_train, y_train)
            acc = m.evaluate(X_test, y_test)["accuracy"]
            results.append({"Learning Rate": rate, "Accuracy": acc})
        df = pd.DataFrame(results)
        st.dataframe(df.set_index("Learning Rate"), use_container_width=True)
        st.line_chart(df.set_index("Learning Rate"))
